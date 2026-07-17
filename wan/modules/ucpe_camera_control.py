"""
UCPE (Unified Camera Positional Encoding) Module for Wan2.2-5B

This module implements the UCPE camera control mechanism from the paper
"Unified Camera Positional Encoding for Controlled Video Generation".

UCPE integrates:
1. Relative Ray Encoding - geometry-consistent alternative to Plücker rays
2. Absolute Orientation Encoding - controllable pitch and roll

Reference: https://arxiv.org/abs/2512.07237
"""

import torch
from torch import nn 
from .prope import PropeDotProductAttention, _invert_SE3
from .attention import flash_attention
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
from typing import Tuple
from .equilib import ucm_unproject_grid, create_grid
from .wan_camera_adapter import SimpleAdapter


def _compute_token_positions(T, memory_len, mem_grid_sizes, patches_x, patches_y,
                              mem_compress_t=1, hr_len=0, hr_grid_sizes=None,
                              device='cpu'):
    """
    Compute per-token x/y/t positions for all tokens (memory + hr + dit).
    Memory positions are scaled to DiT coordinate space.
    HR frame positions use t=-1 with full DiT spatial resolution.

    Args:
        T: total sequence length (memory + hr + dit tokens)
        memory_len: number of memory tokens prepended to the sequence
        mem_grid_sizes: [B, 3] memory grid sizes (mem_F, mem_H, mem_W), or None
        patches_x: DiT patches in x direction (W)
        patches_y: DiT patches in y direction (H)
        mem_compress_t: temporal compression ratio from memory encoder (compress_t)
        hr_len: number of HR frame tokens inserted between memory and dit (0 if none)
        hr_grid_sizes: [B, 3] HR frame grid sizes (1, H_patch, W_patch), or None
        device: torch device

    Returns:
        all_x, all_y, all_t: each shape [T], per-token float positions
            - Memory tokens have scaled coordinates in DiT space
            - Memory t positions are negative (placed before dit t=0)
            - HR frame tokens have t=-1, spatial same as DiT [0, 1, ..., H-1] x [0, 1, ..., W-1]
            - DiT tokens have standard grid coordinates [0, 1, 2, ...]
    """
    H, W = patches_y, patches_x
    dit_tokens = T - memory_len - hr_len
    dit_f = dit_tokens // (H * W) if H * W > 0 else 0

    if memory_len > 0 and mem_grid_sizes is not None:
        mem_f = int(mem_grid_sizes[0, 0].item())
        mem_h = int(mem_grid_sizes[0, 1].item())
        mem_w = int(mem_grid_sizes[0, 2].item())

        # Spatial scaling: memory -> DiT coordinate space
        x_step = W / mem_w  # e.g., 52/26 = 2.0
        y_step = H / mem_h  # e.g., 30/15 = 2.0
        mem_x_coords = torch.arange(mem_w, device=device, dtype=torch.float32) * x_step + (x_step - 1) / 2
        mem_y_coords = torch.arange(mem_h, device=device, dtype=torch.float32) * y_step + (y_step - 1) / 2

        # Build per-token x/y positions for memory: [mem_f * mem_h * mem_w]
        # Token order: frame-major, then row-major (y, x)
        mem_x_per_token = mem_x_coords.repeat(mem_h).repeat(mem_f)
        mem_y_per_token = mem_y_coords.repeat_interleave(mem_w).repeat(mem_f)

        # Temporal scaling: use compress_t from memory encoder
        t_compress_ratio = mem_compress_t  # e.g., 2 for "2x4x4" compression rate
        original_dit_t = mem_f * t_compress_ratio  # pre-compression context frames in DiT space
        # Each memory frame i -> original frame i * t_compress_ratio
        # Placed at negative: -(original_dit_t) + i * t_compress_ratio
        # e.g., mem_f=10, ratio=2: [-20, -18, -16, ..., -2]
        # mem_t_coords = torch.arange(mem_f, device=device, dtype=torch.float32) * t_compress_ratio - original_dit_t
        mem_t_coords = -4000 + torch.arange(mem_f, device=device, dtype=torch.float32) * t_compress_ratio
        mem_t_per_token = mem_t_coords.repeat_interleave(mem_h * mem_w)
    else:
        mem_x_per_token = torch.tensor([], device=device, dtype=torch.float32)
        mem_y_per_token = torch.tensor([], device=device, dtype=torch.float32)
        mem_t_per_token = torch.tensor([], device=device, dtype=torch.float32)

    # HR frame positions: t=-1, spatial same as DiT (full resolution)
    if hr_len > 0 and hr_grid_sizes is not None:
        hr_f = int(hr_grid_sizes[0, 0].item())  # should be 1
        hr_h = int(hr_grid_sizes[0, 1].item())  # same as DiT H
        hr_w = int(hr_grid_sizes[0, 2].item())  # same as DiT W

        # Spatial: same integer positions as DiT [0, 1, ..., W-1] x [0, 1, ..., H-1]
        hr_x_per_token = torch.tile(
            torch.arange(hr_w, device=device, dtype=torch.float32), (hr_h * hr_f,))
        hr_y_per_token = torch.tile(
            torch.repeat_interleave(
                torch.arange(hr_h, device=device, dtype=torch.float32), hr_w),
            (hr_f,))
        # Temporal: fixed at t=-4200
        # hr_t_per_token = torch.full((hr_len,), -1.0, device=device, dtype=torch.float32)
        hr_t_per_token = torch.full((hr_len,), -4200.0, device=device, dtype=torch.float32)
    else:
        hr_x_per_token = torch.tensor([], device=device, dtype=torch.float32)
        hr_y_per_token = torch.tensor([], device=device, dtype=torch.float32)
        hr_t_per_token = torch.tensor([], device=device, dtype=torch.float32)

    # DiT positions: standard grid [0, 1, 2, ..., patches-1]
    dit_x_per_token = torch.tile(
        torch.arange(W, device=device, dtype=torch.float32), (H * dit_f,))
    dit_y_per_token = torch.tile(
        torch.repeat_interleave(
            torch.arange(H, device=device, dtype=torch.float32), W),
        (dit_f,))
    dit_t_per_token = torch.arange(
        dit_f, device=device, dtype=torch.float32).repeat_interleave(H * W)

    # Concatenate: [memory | hr | dit]
    all_x = torch.cat([mem_x_per_token, hr_x_per_token, dit_x_per_token])
    all_y = torch.cat([mem_y_per_token, hr_y_per_token, dit_y_per_token])
    all_t = torch.cat([mem_t_per_token, hr_t_per_token, dit_t_per_token])

    return all_x, all_y, all_t


def patch_dit(model, method, height, width, vae_downscale_factor=8, attn_compress=1, adaptation_method="parallel", enable_camera=True, save_attn_map=False, attn_save_dir="./attn_maps"):
    """
    Patch DiT model with camera control modules.
    
    Args:
        model: WanModel instance
        method: Camera condition method (e.g., "relray", "prope", "recammaster")
        height: Image height in pixels
        width: Image width in pixels
        vae_downscale_factor: VAE spatial downscale factor (default 8)
        attn_compress: Attention dimension compression factor (default 1)
        adaptation_method: How to integrate camera attention ("parallel", "before", "after")
        enable_camera: Whether to actually inject camera modules. If False, only
            computes patches_x/patches_y but skips camera module injection.
        save_attn_map: Whether to save attention maps during inference
        attn_save_dir: Directory to save attention maps
    
    Returns:
        keywords: List of parameter keywords for gradient control, or empty list if camera disabled
    """
    keywords = []
    
    # Always compute patch sizes (needed for memory encoder P_inv alignment)
    patch_factor = vae_downscale_factor * 2
    patches_x = width // patch_factor
    patches_y = height // patch_factor
    model.patches_x = patches_x
    model.patches_y = patches_y
    model.camera_condition = method if enable_camera else "none"
    
    if not enable_camera:
        return keywords
    
    if method.startswith("recam"):
        if method == "recammaster":
            emb_dim = 14
        elif method == "recam_plucker":
            emb_dim = 6
        else:
            raise ValueError(f"Unknown method: {method}")

        dim = model.blocks[0].self_attn.q.weight.shape[0]
        for block in model.blocks:
            block.cam_encoder = nn.Linear(emb_dim, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        keywords.extend(["cam_encoder", "projector", "self_attn"])

    if method == "plucker":
        model.control_adapter = SimpleAdapter(
            model.in_dim_control_adapter,
            model.dim,
            kernel_size=model.patch_size[1:],
            stride=model.patch_size[1:],
            downscale_factor=model.downscale_factor_control_adapter,
        )
        model.control_adapter.conv.weight.data.zero_()
        model.control_adapter.conv.bias.data.zero_()
        for block in model.control_adapter.residual_blocks:
            block.conv2.weight.data.zero_()
            block.conv2.bias.data.zero_()
        keywords = "*"
    elif any(k in method for k in ("gta", "prope", "relray")):
        if "abs" in method:
            if "absc2w" in method or "absray" in method:
                emb_dim = 12
            elif "absmap" in method:
                emb_dim = 3
            else:
                raise ValueError(f"Unknown absolute encoding method: {method}")
        else:
            emb_dim = None

        for block_idx, block in enumerate(model.blocks):
            block.cam_self_attn = UcpeSelfAttention(
                model.dim,
                model.dim // attn_compress,
                block.num_heads // attn_compress,
                patches_x=patches_x,
                patches_y=patches_y,
                image_width=width,
                image_height=height,
                emb_dim=emb_dim,
                adaptation_method=adaptation_method,
                save_attn_map=save_attn_map,
                attn_save_dir=attn_save_dir,
            )
            # Store block index for attention map saving (only save from first block)
            block.cam_self_attn.block_idx = block_idx
            # Also attach prope_attn to self_attn for DiT prope/memrope modes
            block.self_attn.prope_attn = PropeDotProductAttention(
                head_dim=block.self_attn.head_dim,
                patches_x=patches_x,
                patches_y=patches_y,
                image_width=width,
                image_height=height,
                precompute_coeffs=True,
            )
        keywords.append("cam_self_attn")

    return keywords


def enable_grad(model, keywords):
    model.eval()
    model.requires_grad_(False)
    if keywords == "*":
        model.train()
        model.requires_grad_(True)
    else:
        for name, module in model.named_modules():
            if any(keyword in name for keyword in keywords):
                print(f"Trainable: {name}")
                module.train()
                module.requires_grad_(True)

    trainable_params = 0
    seen_params = set()
    for name, module in model.named_modules():
        for param in module.parameters():
            if param.requires_grad and param not in seen_params:
                trainable_params += param.numel()
                seen_params.add(param)
    print(f"Total number of trainable parameters: {trainable_params}")


def compute_fx_from_fov_xi(
    x_fov: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute focal length fx from horizontal FOV and UCM xi."""
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_1d(x_fov)
    xi = to_tensor_1d(xi)

    B = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.view(-1).expand(B)
    xi = xi.view(-1).expand(B)

    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def compute_fov_from_fx_xi(
    fx: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device="cpu",
    dtype=torch.float32,
):
    """Compute horizontal FOV in degrees from UCM fx and xi."""
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx  = to_tensor_1d(fx).view(-1)
    xi  = to_tensor_1d(xi).view(-1)
    B = max(fx.shape[0], xi.shape[0])
    fx  = fx.expand(B)
    xi  = xi.expand(B)

    # A = 2 fx / W
    A = 2.0 * fx / width

    # phi = arctan(1/A)
    phi = torch.atan(1.0 / A)

    # sin(theta - phi) = xi / sqrt(A^2 + 1)
    denom = torch.sqrt(A * A + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi

    # x_fov = 2 * theta (rad → deg)
    x_fov = torch.rad2deg(2.0 * theta)
    return x_fov


def ucm_unproject_grid_fov(
    x_fov: float | torch.Tensor,
    xi: float | torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute UCM camera rays from FOV-based intrinsics."""
    is_batched = any(torch.is_tensor(p) and p.ndim == 1 for p in [x_fov, xi])

    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    fy = fx

    d_cam = ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2,
        xi=xi if torch.is_tensor(xi) else torch.tensor([xi], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        y_down=True,
    )

    if not is_batched:
        d_cam = d_cam[0]  # [H, W, 3]

    return d_cam


def project_ucm_points_fov(X, Y, Z, x_fov, xi, height, width):
    """
    Project 3D points in camera frame to UCM image plane using fov-based intrinsics.

    Args:
        X, Y, Z: torch.Tensor [..., 3D coordinates in camera frame]
        x_fov: float or [B] —— horizontal field of view in degrees
        xi: float or [B] —— UCM mirror parameter
        height, width: target image dimensions

    Returns:
        du, dv: projected pixel coordinates [..., 2]
    """
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    fy = fx
    cx = width / 2
    cy = height / 2

    return project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi)


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    """
    Project 3D points in camera frame to UCM image plane.

    Args:
        X, Y, Z: torch.Tensor [..., 3D coordinates in camera frame]
        fx, fy, cx, cy: intrinsics (scalars or tensors)
        xi: UCM mirror parameter

    Returns:
        du, dv: projected pixel coordinates [..., 2]
    """
    r = torch.sqrt(X * X + Y * Y + Z * Z)
    alpha = Z + xi.view(-1, 1, 1, 1) * r
    du = fx.view(-1, 1, 1, 1) * (X / alpha) + cx
    dv = fy.view(-1, 1, 1, 1) * (Y / alpha) + cy
    return du, dv


def ray_condition_ucm(
    x_fov,      # float or [B] —— same fov as used in equi2pers
    xi,        # float or [B] —— same xi as used in equi2pers
    pose,       # [B, V, 4, 4]
    height, width,      # target height, width
    device,
):
    """
    ✅ UCM-based Plücker embedding, output format: [B, V, H, W, 6]
    🔁 Internally uses your ucm_unproject_grid() for consistent ray geometry.
    
    Only required params:
        fov_x  (degree)
        xi
        c2w    (camera-to-world pose, same as your exported pose)
        H, W   (spatial resolution)
        device
    """

    d_cam = ucm_unproject_grid_fov(
        x_fov, xi, height, width, device, dtype=pose.dtype
    )
    d_cam = repeat(d_cam, "b ... -> b v ...", v=pose.shape[1])  # [B, V, H, W, 3]
    mask = d_cam.isnan().any(-1)

    # --- 4. Transform rays into world coordinates using c2w ---
    R = pose[..., :3, :3]      # [B, V, 3, 3]
    t = pose[..., :3, 3]       # [B, V, 3]

    d_world = torch.einsum("bvij,bvhwj->bvhwi", R.transpose(-1, -2), d_cam)  # [B,V,H,W,3]
    rays_o = t[..., None, None, :].expand_as(d_world)  # [B,V,H,W,3]

    # --- 5. Plücker coordinates: m = o × d ---
    m = torch.cross(rays_o, d_world, dim=-1)  # [B,V,H,W,3]

    # --- 6. Final concat: [m, d] → [B,V,H,W,6]
    plucker = torch.cat([m, d_world], dim=-1)
    plucker[mask] = 0.
    return plucker


def d_cam_to_angles(d_cam: torch.Tensor) -> torch.Tensor:
    """Convert camera directions to azimuth/elevation angles in radians."""
    d_unit = F.normalize(d_cam, dim=-1)  # [B, H, W, 3]

    x = d_unit[..., 0]  # right
    y = d_unit[..., 1]  # down
    z = d_unit[..., 2]  # forward

    # yaw / azimuth: angle in xz-plane
    azimuth = torch.atan2(x, z)  # ∈ [-π, π]

    # pitch / elevation: angle above xz-plane
    elevation = -torch.asin(y)   # y is downward

    return torch.stack([azimuth, elevation], dim=-1)  # [B, H, W, 2]


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [B, H, W, 3]
    c2w: torch.Tensor,    # [B, T, 4, 4]
) -> torch.Tensor:
    """Build world-to-ray local transforms for each ray."""
    B, H, W, _ = d_cam.shape
    T = c2w.shape[1]
    device = d_cam.device
    dtype = d_cam.dtype

    # --- Expand ray dirs across frames ---
    # [B,H,W,3] -> [B,T,H,W,3]
    d_cam = repeat(d_cam, 'b h w c -> b t h w c', t=T)

    # extract camera R,t
    R_cam = c2w[..., :3, :3]       # [B,T,3,3]
    t_cam = c2w[..., :3, 3]        # [B,T,3]
    
    # --- d_world: rotate ray directions into world ---
    d_world = einsum(R_cam, d_cam, 'b t i j, b t h w j -> b t h w i')

    # camera y-axis from each view
    cam_y = R_cam[..., :, 1]       # [B,T,3]
    cam_y = repeat(cam_y, 'b t c -> b t h w c', h=H, w=W)

    # Orthonormal ray-local axes
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)
    
    # local->world rotation
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)  # [B,T,H,W,3,3]

    # world->local rotation (transpose)
    R_w2l = rearrange(R_l2w, 'b t h w i j -> b t h w j i')  # ✅

    # broadcast camera center
    t_world = repeat(t_cam, 'b t c -> b t h w c', h=H, w=W)

    # world->local translation
    t_w2l = -einsum(R_w2l, t_world, 'b t h w i j, b t h w j -> b t h w i')

    # assemble transform matrix
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0

    # NaN handling
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)

    return raymats


def rope_precompute_coeffs(
    positions: torch.Tensor,  # [B, H, W]
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, 1, H*W, D], [B, 1, H*W, D]
    """Precompute RoPE cosine and sine coefficients for ray-angle positions."""
    mask = positions.isnan()
    positions = positions.clone()
    positions[mask] = 0.0

    B, H, W = positions.shape
    positions_flat = positions.view(B, H * W)  # [B, HW]
    num_freqs = feat_dim // 2

    freqs = freq_scale * (
        freq_base ** (
            -torch.arange(num_freqs, device=positions.device)[None, :]
            / num_freqs
        )  # [1, D]
    )  # [1, D]

    # Expand for batch & positions
    angles = positions_flat[..., None] * freqs[None, :, :]  # [B, HW, D]
    angles = angles.view(B, 1, H * W, num_freqs)

    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def compute_up_lat_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
):
    """Compute up and latitude maps for camera rays."""
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()

    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=height,
        width=width,
        device=device,
        dtype=torch.float32,
    )  # [B, H, W, 3]
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)  # [B, H, W, 3]
    mask = d_cam.isnan().any(dim=-1, keepdim=True)  # [B, H, W, 1]

    d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)  # [B, T, H, W, 3]
    d_world = torch.einsum('btij,bthwj->bthwi', R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)

    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)  # [B, T, H, W, 1]

    v = d_world
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = torch.cross(v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)

    delta = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta)
    sin_eps = torch.sin(delta)
    v_rot = v * cos_eps + torch.cross(k, v, dim=-1) * sin_eps + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)

    dirs_cam = torch.einsum('btij,bthwj->bthwi', R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]

    du, dv = project_ucm_points_fov(
        Xs, Ys, Zs,
        x_fov=x_fov.float(),
        xi=xi.float(),
        height=height,
        width=width,
    )
    grid = create_grid(
        height=height,
        width=width,
        batch=B,
        dtype=torch.float32,
        device=device,
    )  # [B, H, W, 3]
    grid_x = grid[..., 0].unsqueeze(1)  # [B,1,H,W]
    grid_y = grid[..., 1].unsqueeze(1)

    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)  # [B, T, H, W, 2]
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)

    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)

    mask_exp2 = mask.unsqueeze(1).expand(B, T, height, width, 1)
    up_map = up_map.masked_fill(mask_exp2, 0.0)
    lat_map = lat_map.masked_fill(mask_exp2, 0.0)

    return up_map, lat_map


class UcpeSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_dim: int,
        num_heads: int,
        patches_x: int = 8,
        patches_y: int = 8,
        image_width: int = 128,
        image_height: int = 128,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        precompute_coeffs: bool = True,
        emb_dim: int | None = None,
        adaptation_method: str = "parallel",
        save_attn_map: bool = False,
        attn_save_dir: str = "./attn_maps",
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        self.adaptation_method = adaptation_method
        self.save_attn_map = save_attn_map
        self.attn_save_dir = attn_save_dir

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        if emb_dim is not None:
            self.cam_encoder = nn.Linear(emb_dim, dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.prope_attn = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            freq_base=freq_base,
            freq_scale=freq_scale,
            precompute_coeffs=precompute_coeffs,
        )

    def forward(self, x: torch.Tensor, control_camera_dit_input: dict,
                rope_mode: str = 'rope', memory_len: int = 0,
                mem_grid_sizes=None, mem_compress_t: int = 1,
                save_attn_step: int = None, hr_len: int = 0):
        """
        Args:
            x: (B, T, D) — input tokens (may include memory tokens prepended)
            control_camera_dit_input: dict with keys:
                - viewmats: (B, N, 4, 4) — may include memory + dit viewmats concatenated
                - K: (B, N, 3, 3) — optional
                - save_attn_step: If not None, save attention map at this denoising step
                - seg_idx: Segment index for filename distinction
            rope_mode: Position encoding mode ('rope', 'prope', 'memrope')
            memory_len: Number of memory tokens prepended to x (0 if none)
            mem_grid_sizes: [B, 3] memory grid sizes (mem_F, mem_H, mem_W), needed for memrope
            mem_compress_t: temporal compression ratio from memory encoder (default 1)
            save_attn_step: If not None, save attention map at this denoising step (deprecated, use dict)
            hr_len: Number of HR frame tokens inserted between memory and dit (0 if none)
        """
        B, T, D = x.shape
        N = control_camera_dit_input["viewmats"].shape[1]  # number of viewmat entries
        H, W = self.patches_y, self.patches_x
        
        # Extract seg_idx from control_camera_dit_input if present
        seg_idx = control_camera_dit_input.get('seg_idx', 0) if isinstance(control_camera_dit_input, dict) else 0

        # cam_emb additive embedding: only used in rope mode (relray_absmap etc.)
        # In prope/memrope modes, camera info is encoded via PRoPE rotary encoding,
        # so cam_emb additive embedding is redundant and should be skipped.
        if hasattr(self, "cam_encoder") and "cam_emb" in control_camera_dit_input and rope_mode == 'rope':
            cam_emb = control_camera_dit_input["cam_emb"]
            y = self.cam_encoder(cam_emb)
            if memory_len > 0:
                # cam_emb may contain [mem_cam_emb | dit_cam_emb] concatenated,
                # but memory cam_emb has different spatial resolution than memory tokens.
                # Extract only the dit portion of cam_emb and apply to dit tokens.
                # Memory tokens get zero cam_emb (camera info encoded via viewmats/PRoPE).
                dit_tokens = T - memory_len
                dit_f = dit_tokens // (H * W) if H * W > 0 else 0

                # The dit cam_emb is the last dit_f*H*W entries (or last dit_f per-frame entries)
                if y.shape[1] == T:
                    # cam_emb already matches total tokens (unlikely but handle)
                    pass
                else:
                    # Extract dit portion: last dit_f entries (per-frame) or last dit_f*H*W (per-token)
                    if y.shape[1] >= dit_f * H * W:
                        # Per-token cam_emb: take last dit_f*H*W
                        dit_y = y[:, -dit_f * H * W:]
                    elif y.shape[1] >= dit_f:
                        # Per-frame cam_emb: take last dit_f and repeat
                        dit_y = y[:, -dit_f:]
                        dit_y = repeat(dit_y, "b f d -> b (f hw) d", hw=H * W)
                    else:
                        dit_y = y
                        hw = dit_tokens // y.shape[1] if y.shape[1] > 0 else 0
                        if hw > 0:
                            dit_y = repeat(dit_y, "b f d -> b (f hw) d", hw=hw)

                    # Pad memory portion with zeros
                    mem_y = torch.zeros(y.shape[0], memory_len, dit_y.shape[2],
                                       device=y.device, dtype=y.dtype)
                    y = torch.cat([mem_y, dit_y], dim=1)
            elif y.shape[1] != T:
                hw = T // y.shape[1]
                y = repeat(y, "b f d -> b (f hw) d", hw=hw)
            x = x + y

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_head]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_mode == 'rope':
            # Standard RoPE mode: apply complex-frequency x/y/t RoPE (no camera projection)
            from wan.modules.prope import _rope_precompute_coeffs

            # head_dim split for standard RoPE: d_t + d_h + d_w = head_dim
            c = self.head_dim // 2
            # d_h and d_w must be even for RoPE (cos/sin pairs)
            d_h = (c // 3) // 2 * 2  # round down to even
            d_w = (c // 3) // 2 * 2  # round down to even
            d_t = c - d_h - d_w  # remainder goes to temporal

            # Compute real-coordinate positions for all tokens
            all_x, all_y, all_t = _compute_token_positions(
                T=T, memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                patches_x=W, patches_y=H, mem_compress_t=mem_compress_t,
                device=x.device,
            )

            # Precompute RoPE coefficients for t, y, x
            coeffs_t = _rope_precompute_coeffs(
                all_t, freq_base=10000.0, freq_scale=1.0, feat_dim=d_t, dtype=x.dtype)
            coeffs_y = _rope_precompute_coeffs(
                all_y, freq_base=10000.0, freq_scale=1.0, feat_dim=d_h, dtype=x.dtype)
            coeffs_x = _rope_precompute_coeffs(
                all_x, freq_base=10000.0, freq_scale=1.0, feat_dim=d_w, dtype=x.dtype)

            # Apply RoPE: split head_dim into [t, y, x] and apply rotary to each
            # coeffs are (cos, sin) tuples, each shape [T, d//2]
            def _apply_rope_3d(tensor, coeffs_t, coeffs_y, coeffs_x, d_t, d_h, d_w):
                """Apply 3D RoPE (t, y, x) to tensor [B, H, T, D]."""
                # Split into t, y, x portions
                t_part = tensor[..., :d_t]
                y_part = tensor[..., d_t:d_t + d_h]
                x_part = tensor[..., d_t + d_h:d_t + d_h + d_w]
                rest = tensor[..., d_t + d_h + d_w:]  # remaining dims unchanged

                def _rotate(x_in, cos_sin):
                    cos, sin = cos_sin  # each [1, 1, seqlen, num_freqs] from _rope_precompute_coeffs
                    # coeffs already have shape [1, 1, T, d//2], broadcast with x_in [B, H, T, d]
                    d = x_in.shape[-1]
                    x1 = x_in[..., :d // 2]
                    x2 = x_in[..., d // 2:]
                    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

                t_rot = _rotate(t_part, coeffs_t) if d_t > 0 else t_part
                y_rot = _rotate(y_part, coeffs_y) if d_h > 0 else y_part
                x_rot = _rotate(x_part, coeffs_x) if d_w > 0 else x_part

                return torch.cat([t_rot, y_rot, x_rot, rest], dim=-1)

            q = _apply_rope_3d(q, coeffs_t, coeffs_y, coeffs_x, d_t, d_h, d_w)
            k = _apply_rope_3d(k, coeffs_t, coeffs_y, coeffs_x, d_t, d_h, d_w)

            # Rearrange to [B, T, H, D] for flash_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Save attention map if requested (for rope mode)
            if self.save_attn_map and save_attn_step is not None:
                self._save_attention_map(q, k, memory_len, T, save_attn_step, "rope", block_idx=getattr(self, 'block_idx', None), seg_idx=seg_idx, hr_len=hr_len, mem_grid_sizes=mem_grid_sizes, mem_compress_t=mem_compress_t)

            out = flash_attention(q, k, v)

            # Final projection
            out = out.flatten(2)
            return self.out_proj(out)

        # prope / memrope modes: use PRoPE camera projection
        # Compute RoPE coefficients for memrope mode (x, y, t with real coordinates)
        coeffs_t = None
        coeffs_x_override = None
        coeffs_y_override = None
        if rope_mode == 'memrope':
            from wan.modules.prope import _rope_precompute_coeffs
            d_x = (self.head_dim // 6) // 2 * 2  # round down to even
            d_y = (self.head_dim // 6) // 2 * 2  # round down to even
            d_t = self.head_dim - self.head_dim // 2 - d_x - d_y

            # Compute real-coordinate positions for all tokens using shared utility
            all_x_positions, all_y_positions, all_t_positions = _compute_token_positions(
                T=T, memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                patches_x=W, patches_y=H, mem_compress_t=mem_compress_t,
                device=x.device,
            )

            # Precompute coeffs for x, y, t
            coeffs_x_override = _rope_precompute_coeffs(
                all_x_positions, freq_base=100.0, freq_scale=1.0, feat_dim=d_x, dtype=x.dtype,
            )
            coeffs_y_override = _rope_precompute_coeffs(
                all_y_positions, freq_base=100.0, freq_scale=1.0, feat_dim=d_y, dtype=x.dtype,
            )
            coeffs_t = _rope_precompute_coeffs(
                all_t_positions, freq_base=100.0, freq_scale=1.0, feat_dim=d_t, dtype=x.dtype,
            )

        # Precompute camera-specific functions
        # Ensure viewmats dtype matches input x to avoid einsum dtype mismatch (e.g., float32 vs bfloat16)
        viewmats = control_camera_dit_input["viewmats"].to(dtype=x.dtype)
        Ks = control_camera_dit_input.get("K", None)
        if Ks is not None:
            Ks = Ks.to(dtype=x.dtype)
        self.prope_attn._precompute_and_cache_apply_fns(
            viewmats=viewmats,
            Ks=Ks,
            coeffs_x=coeffs_x_override if coeffs_x_override is not None else control_camera_dit_input.get("coeffs_x", None),
            coeffs_y=coeffs_y_override if coeffs_y_override is not None else control_camera_dit_input.get("coeffs_y", None),
            rope_mode=rope_mode,
            coeffs_t=coeffs_t,
        )

        # Apply RoPE-style positional encoding
        q = self.prope_attn._apply_to_q(q)     # [B, H, T, D_head]
        k = self.prope_attn._apply_to_kv(k)
        v = self.prope_attn._apply_to_kv(v)

        # Save attention map if requested (for prope/memrope mode)
        if self.save_attn_map and save_attn_step is not None:
            self._save_attention_map(q, k, memory_len, T, save_attn_step, rope_mode, block_idx=getattr(self, 'block_idx', None), seg_idx=seg_idx, hr_len=hr_len, mem_grid_sizes=mem_grid_sizes, mem_compress_t=mem_compress_t)

        # Rearrange to [B, T, D] for flash_attention input
        q = rearrange(q, "b h t d -> b t (h d)")
        k = rearrange(k, "b h t d -> b t (h d)")
        v = rearrange(v, "b h t d -> b t (h d)")

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        # Fast attention (Flash/Sage/SDPA fallback)
        out = flash_attention(q, k, v)

        # reshape back
        out = out.view(B, T, self.num_heads, self.head_dim)
        out = rearrange(out, "b t h d -> b h t d")

        # Apply inverse transform for PRoPE
        out = self.prope_attn._apply_to_o(out)
        # Final projection
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)

    def _save_attention_map(self, q, k, memory_len, T, save_attn_step, mode="rope",
                            block_idx=None, seg_idx=0, hr_len=0, mem_grid_sizes=None,
                            mem_compress_t: int = 1):
        """
        Save attention map for current dit tokens only.
        
        Args:
            q: Query tensor [B, num_heads, T, head_dim] or [B, T, num_heads, head_dim]
            k: Key tensor [B, num_heads, T, head_dim] or [B, T, num_heads, head_dim]
            memory_len: Number of memory tokens
            T: Total sequence length
            save_attn_step: Current denoising step
            mode: Attention mode ('rope', 'prope', 'memrope')
            block_idx: Block index for saving (save every 4th block: 0, 4, 8, ...)
            seg_idx: Segment index for filename distinction
            hr_len: Number of HR frame tokens inserted between memory and dit (0 if none)
            mem_grid_sizes: [B, 3] memory grid sizes (mem_F, mem_H, mem_W), used to
                compute memory_tokens_per_frame for visualization pooling
            mem_compress_t: temporal compression ratio from memory encoder, used to
                upsample memory columns in visualization to match dit frame count
        """
        import os
        import numpy as np
        
        # Save from every 4th block to get 8 layers from 32-layer model
        # Blocks: 0, 4, 8, 12, 16, 20, 24, 28
        if block_idx is not None and block_idx % 4 != 0:
            return
        
        with torch.no_grad():
            # Handle different tensor layouts
            if q.dim() == 4:
                if q.shape[1] == self.num_heads:
                    # Layout: [B, num_heads, T, head_dim]
                    q_for_attn = q
                    k_for_attn = k
                else:
                    # Layout: [B, T, num_heads, head_dim] -> transpose to [B, num_heads, T, head_dim]
                    q_for_attn = q.transpose(1, 2)
                    k_for_attn = k.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected q dimension: {q.dim()}")
            
            B, num_heads, total_T, head_dim = q_for_attn.shape
            
            # Compute attention scores (QK dot product before softmax)
            scale = head_dim ** 0.5
            attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_for_attn / scale, k_for_attn)
            
            # Extract only dit tokens' rows (queries) for scores
            # Token layout: [memory (N_mem) | hr (N_hr) | dit (rest)]
            context_len = memory_len + hr_len
            if context_len > 0 and context_len < total_T:
                dit_start = context_len
                attn_scores_dit = attn_scores[:, :, dit_start:, :]  # [B, num_heads, dit_tokens, T]
            else:
                attn_scores_dit = attn_scores
            
            # Average over heads and batch for scores
            attn_scores_avg = attn_scores_dit.mean(dim=1).mean(dim=0)  # [dit_tokens, T]
            
            # Save QK scores to file (full resolution, no pooling)
            os.makedirs(self.attn_save_dir, exist_ok=True)
            
            # Convert to float32 for numpy compatibility (numpy doesn't support bfloat16)
            attn_scores_np = attn_scores_avg.cpu().float().numpy()
            
            # Save QK scores (before softmax)
            scores_path = os.path.join(self.attn_save_dir, f"attn_seg{seg_idx}_step_{save_attn_step}_block_{block_idx}_mode_{mode}.npy")
            np.save(scores_path, attn_scores_np)
            
            # Build and save token type metadata
            dit_len = total_T - memory_len - hr_len
            token_types = np.zeros(total_T, dtype=np.int32)
            if memory_len > 0:
                token_types[:memory_len] = 0  # memory
            if hr_len > 0:
                token_types[memory_len:memory_len + hr_len] = 1  # hr
            token_types[memory_len + hr_len:] = 2  # dit
            
            # Compute tokens-per-frame for each token type (for latent-aware visualization pooling)
            dit_tokens_per_frame = self.patches_x * self.patches_y
            
            # Memory tokens per frame: mem_h * mem_w from mem_grid_sizes
            memory_tokens_per_frame = 0
            if mem_grid_sizes is not None and memory_len > 0:
                mem_h = int(mem_grid_sizes[0, 1].item())
                mem_w = int(mem_grid_sizes[0, 2].item())
                memory_tokens_per_frame = mem_h * mem_w
            
            # HR tokens per frame: HR is always 1 frame, so tokens_per_frame = hr_len
            hr_tokens_per_frame = hr_len if hr_len > 0 else 0
            
            meta_path = scores_path.replace('.npy', '_meta.npz')
            np.savez(meta_path,
                     memory_len=memory_len,
                     hr_len=hr_len,
                     dit_len=dit_len,
                     total_T=total_T,
                     token_types=token_types,
                     memory_start=0,
                     memory_end=memory_len,
                     hr_start=memory_len,
                     hr_end=memory_len + hr_len,
                     dit_start=memory_len + hr_len,
                     dit_end=total_T,
                     dit_tokens_per_frame=dit_tokens_per_frame,
                     memory_tokens_per_frame=memory_tokens_per_frame,
                     hr_tokens_per_frame=hr_tokens_per_frame,
                     mem_compress_t=mem_compress_t)
            
            print(f"[UCPE Attn] Saved QK scores to {scores_path}, shape: {attn_scores_avg.shape} "
                  f"(memory={memory_len}, hr={hr_len}, dit={dit_len}, total={total_T})")

def prepare_camera_input(
    pose: torch.Tensor,  # [B, T, 3, 4] or [B, T, 4, 4]
    x_fov: torch.Tensor,  # [B]
    xi: torch.Tensor,  # [B]
    method: str,
    patches_x: int,
    patches_y: int,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:

    if method == "recammaster":
        pose_emb = rearrange(pose[:, ::4], "B T ... -> B T (...)")
        xi_emb = repeat(xi, "B -> B T 1", T=pose_emb.shape[1])
        x_fov_emb = repeat(x_fov / 180, "B -> B T 1", T=pose_emb.shape[1])
        cam_emb = torch.cat([pose_emb, xi_emb, x_fov_emb], dim=-1)
        return {"cam_emb": cam_emb}
    elif method == "recam_plucker":
        pose = pose[:, ::4]  # shape [B, T, 3, 4]
        plucker_ucm = ray_condition_ucm(
            x_fov=x_fov,
            xi=xi,
            pose=pose,
            height=patches_y,
            width=patches_x,
            device=device,
        )
        plucker_ucm = rearrange(plucker_ucm, "B T H W C -> B (T H W) C")
        return {"cam_emb": plucker_ucm}
    elif method == "plucker":
        plucker_ucm = ray_condition_ucm(
            x_fov=x_fov,
            xi=xi,
            pose=pose,
            height=height,
            width=width,
            device=device,
        )

        control_camera_video = plucker_ucm.permute(0, 4, 1, 2, 3)  # [1, 6, V, H, W]

        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ],
            dim=2
        ).transpose(1, 2)  # → shape now [1, C', V, H, W]

        b, f, c, h, w = control_camera_latents.shape
        assert f % 4 == 0, f"Expected frame count divisible by 4, but got {f}"

        control_camera_latents = control_camera_latents.contiguous().view(
            b, f // 4, 4, c, h, w
        ).transpose(2, 3)  # → [1, f//4, c, 4, H, W]

        control_camera_latents = control_camera_latents.contiguous().view(
            b, f // 4, c * 4, h, w
        ).transpose(1, 2)  # → final [1, C_out, f//4, H, W]

        control_camera_latents_input = control_camera_latents.to(
            device=device, dtype=dtype
        )
        return control_camera_latents_input
    else:
        pose = pose[:, ::4]  # [B, T//4, 3, 4]
        # Ensure pose is 4x4
        if pose.shape[-2] == 3:
            last_row = torch.tensor([[0, 0, 0, 1]], device=pose.device, dtype=pose.dtype)
            last_row = last_row.expand(pose.shape[0], pose.shape[1], 1, 4)
            c2w = torch.cat([pose, last_row], dim=-2)
        else:
            c2w = pose
        
        if "gta" in method or "prope" in method:
            viewmats = _invert_SE3(c2w)
        elif "relray" in method:
            d_cam = ucm_unproject_grid_fov(
                x_fov=x_fov,
                xi=xi,
                height=patches_y,
                width=patches_x,
                device=device,
                dtype=pose.dtype,
            ) # [B, H, W, 3]
            raymats = world_to_ray_mats(d_cam, c2w)  # [B, T, H, W, 4, 4]
            viewmats = rearrange(raymats, "B T H W ... -> B (T H W) ...") # [B, TxHxW, 4, 4]
        else:
            raise ValueError(f"Unknown camera condition method: {method}")

        control_camera_input = {
            "viewmats": viewmats,
        }

        if method == "prope":
            theta = torch.deg2rad(0.5 * x_fov)  # [B,V]
            eps = torch.finfo(pose.dtype).eps
            denom = torch.sin(theta).clamp_min(eps)
            fx = (width * 0.5) * (torch.cos(theta) + xi) / denom  # [B,V]
            fy = fx  # square pixel assumption
            Ks = torch.zeros((pose.shape[0], pose.shape[1], 3, 3), device=pose.device, dtype=pose.dtype)  # [B,V,3,3]
            Ks[..., 0, 0] = fx.unsqueeze(-1)
            Ks[..., 1, 1] = fy.unsqueeze(-1)
            Ks[..., 0, 2] = width * 0.5
            Ks[..., 1, 2] = height * 0.5
            Ks[..., 2, 2] = 1.0
            control_camera_input["K"] = Ks

        if "absc2w" in method:
            cam_emb = rearrange(pose[:, :, :3, :], "B T ... -> B T (...)")
            control_camera_input["cam_emb"] = cam_emb
        elif "absray" in method:
            cam_emb = viewmats[..., :3, :]
            control_camera_input["cam_emb"] = rearrange(cam_emb, "B N ... -> B N (...)")
        elif "absmap" in method:
            up_map, lat_map = compute_up_lat_map(
                R=pose[..., :3, :3],  # [B, T, 3, 3]
                x_fov=x_fov,
                xi=xi,
                height=patches_y,
                width=patches_x,
                device=device,
            )
            absmap = torch.cat([up_map, lat_map], dim=-1)
            cam_emb = rearrange(absmap, "B T H W C -> B (T H W) C")
            control_camera_input["cam_emb"] = cam_emb

        return control_camera_input