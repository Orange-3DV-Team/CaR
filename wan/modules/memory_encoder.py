"""
Memory Compression Encoder for Wan2.2-TI2V-5B

Based on the paper: "Pretraining Frame Preservation in Autoregressive Video Memory Compression"

This module implements a memory compression encoder that compresses long video latents
into short context features for autoregressive video generation with camera control.

Enhanced with UCPE (Unified Camera Position Encoding) for camera-aware compression.
Supports relray mode for per-patch ray-local coordinate system transforms.

Adapted from UCPE_memory_pfp_t2v/models/new_memory_compression_encoder.py
for Wan2.2-TI2V-5B (dim=2048, num_heads=16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple, Optional, Dict, List
import math
import logging

# Import PRoPE attention for UCPE camera encoding
from .prope import PropeDotProductAttention, _prepare_apply_fns, _rope_precompute_coeffs

# Import relray mode functions for per-patch ray transforms
from .ucpe_camera_control import ucm_unproject_grid_fov, world_to_ray_mats

# Import flash attention for efficient attention computation
from .attention import flash_attention

logger = logging.getLogger(__name__)


def vae_style_temporal_downsample(tensor, dim=1):
    """
    VAE-style temporal downsampling: repeat first frame 4 times + remaining frames,
    then take every 4th frame.
    
    Args:
        tensor: Input tensor, temporal dimension at `dim`
        dim: Temporal dimension index, default 1 (for [B, T, ...] format)
    
    Returns:
        Downsampled tensor
    """
    first = tensor.narrow(dim, 0, 1)
    first_repeated = first.repeat_interleave(4, dim=dim)
    remaining = tensor.narrow(dim, 1, tensor.shape[dim] - 1)
    expanded = torch.cat([first_repeated, remaining], dim=dim)
    indices = torch.arange(0, expanded.shape[dim], 4, device=tensor.device)
    return expanded.index_select(dim, indices)


def compute_relray_viewmats(
    c2w: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    patches_y: int,
    patches_x: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute relray-style viewmats for per-patch ray-local coordinate system transforms.
    
    Args:
        c2w: [B, T, 4, 4] camera-to-world matrices
        x_fov: [B] or scalar, horizontal field of view in degrees
        xi: [B] or scalar, UCM mirror parameter
        patches_y: number of patches in height
        patches_x: number of patches in width
        device: target device
        dtype: target dtype
    
    Returns:
        viewmats: [B, T*H*W, 4, 4] per-patch ray transformation matrices
    """
    B, T = c2w.shape[:2]
    
    # Ensure c2w matches target dtype to avoid einsum dtype mismatch
    c2w = c2w.to(dtype=dtype)
    
    # Compute ray directions for each patch
    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=patches_y,
        width=patches_x,
        device=device,
        dtype=dtype,
    )
    
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)  # [B, H, W, 3]
    
    # Compute world-to-ray transformation matrices
    raymats = world_to_ray_mats(d_cam, c2w)  # [B, T, H, W, 4, 4]
    
    # Flatten to [B, T*H*W, 4, 4]
    viewmats = rearrange(raymats, "B T H W ... -> B (T H W) ...")
    
    return viewmats


def scale_pixels_for_memory_compression(pixels, compression_rate, vae_spatial_compress=8):
    """
    Scale context pixel video in image space so that the resulting VAE latent
    dimensions are divisible by the memory compression rate.

    Should be called BEFORE VAE encoding for best interpolation quality.
    When used, the Memory Encoder's internal latent-space scaling (fallback)
    will be a no-op since the VAE latent already has divisible dimensions.

    Args:
        pixels: [C, T, H, W] or [B, C, T, H, W] pixel video tensor
        compression_rate: str like "2x8x8" (TxHxW format)
        vae_spatial_compress: VAE spatial compression ratio (default 8)

    Returns:
        Scaled pixel video, same format as input. Returns input unchanged
        if dimensions are already divisible.
    """
    parts = compression_rate.lower().split('x')
    compress_h, compress_w = int(parts[1]), int(parts[2])

    H, W = pixels.shape[-2], pixels.shape[-1]
    target_divisor_h = vae_spatial_compress * compress_h
    target_divisor_w = vae_spatial_compress * compress_w

    if H % target_divisor_h == 0 and W % target_divisor_w == 0:
        return pixels  # No scaling needed

    target_h = math.ceil(H / target_divisor_h) * target_divisor_h
    target_w = math.ceil(W / target_divisor_w) * target_divisor_w

    T = pixels.shape[-3]
    need_unsqueeze = pixels.ndim == 4
    if need_unsqueeze:
        pixels = pixels.unsqueeze(0)  # [1, C, T, H, W]

    pixels = F.interpolate(pixels, size=(T, target_h, target_w), mode='trilinear', align_corners=False)

    if need_unsqueeze:
        pixels = pixels.squeeze(0)  # [C, T, target_H, target_W]

    logger.info(f"Pixel-space scaling for memory compression: ({H}, {W}) -> ({target_h}, {target_w})")
    return pixels


def _get_num_groups(num_channels: int, preferred_groups: int = 32) -> int:
    """Get a valid num_groups for GroupNorm that divides num_channels."""
    if num_channels % preferred_groups == 0:
        return preferred_groups
    # Try common group sizes in descending order
    for g in [32, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1


class ResnetBlock3D(nn.Module):
    """3D ResNet block with GroupNorm."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        ng_in = _get_num_groups(in_channels, num_groups)
        ng_out = _get_num_groups(out_channels, num_groups)
        
        self.norm1 = nn.GroupNorm(num_groups=ng_in, num_channels=in_channels, eps=eps)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
        self.norm2 = nn.GroupNorm(num_groups=ng_out, num_channels=out_channels, eps=eps)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
        self.act = nn.SiLU()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class DownsampleBlock3D(nn.Module):
    """3D Downsampling block with configurable stride."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (2, 2, 2),
        num_groups: int = 32,
    ):
        super().__init__()
        self.resnet = ResnetBlock3D(in_channels, out_channels, num_groups=num_groups)
        self.downsample = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.downsample(x)
        return x


class SelfAttention3D(nn.Module):
    """
    3D Self-Attention with optional UCPE (Unified Camera Position Encoding).
    
    This implementation follows the same pattern as UcpeSelfAttention in
    wan/modules/ucpe_camera_control.py to ensure compatibility.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        patches_x: int = 8,
        patches_y: int = 8,
        image_width: int = 128,
        image_height: int = 128,
        mem_enc_use_ucpe: bool = True,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mem_enc_use_ucpe = mem_enc_use_ucpe
        self.patches_x = patches_x
        self.patches_y = patches_y
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        
        # Initialize output projection to zero for stable training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
        if mem_enc_use_ucpe:
            self.prope_attn = PropeDotProductAttention(
                head_dim=self.head_dim,
                patches_x=patches_x,
                patches_y=patches_y,
                image_width=image_width,
                image_height=image_height,
                freq_base=freq_base,
                freq_scale=freq_scale,
                precompute_coeffs=True,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        camera_params: Optional[Dict] = None,
        spatial_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] where N = T * H * W
            camera_params: dict with camera parameters for UCPE encoding
                - 'viewmats': [B, N, 4, 4] per-patch ray transformation matrices
                - 'K': [B, N, 3, 3] camera intrinsics (optional)
                - 'coeffs_x': precomputed x coefficients (optional)
                - 'coeffs_y': precomputed y coefficients (optional)
            spatial_shape: (T, H, W) actual spatial dimensions for dynamic prope update
        
        Returns:
            Output tensor [B, N, C] with residual connection
        """
        B, N, C = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Project Q, K, V
        q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply UCPE camera encoding if available
        if self.mem_enc_use_ucpe and camera_params is not None and 'viewmats' in camera_params:
            # Dynamically update prope_attn patches to match actual input size
            if spatial_shape is not None:
                T_out, H_out, W_out = spatial_shape
                self.prope_attn.patches_x = W_out
                self.prope_attn.patches_y = H_out
            
            # Precompute camera-specific functions
            self.prope_attn._precompute_and_cache_apply_fns(
                viewmats=camera_params['viewmats'],
                Ks=camera_params.get('K', None),
                coeffs_x=camera_params.get('coeffs_x', None),
                coeffs_y=camera_params.get('coeffs_y', None),
            )
            
            # Apply RoPE-style positional encoding
            q = self.prope_attn._apply_to_q(q)
            k = self.prope_attn._apply_to_kv(k)
            v = self.prope_attn._apply_to_kv(v)
        
        # Use flash attention
        # Rearrange for flash_attention: [B, H, N, D] -> [B, N, H, D]
        q = q.transpose(1, 2)  # [B, N, H, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = flash_attention(q, k, v)
        
        # Apply inverse transform for UCPE
        if self.mem_enc_use_ucpe and camera_params is not None and 'viewmats' in camera_params:
            out = out.transpose(1, 2)  # [B, H, N, D]
            out = self.prope_attn._apply_to_o(out)
            out = out.transpose(1, 2)  # [B, N, H, D]
        
        # Reshape and project
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)
        
        return x + out


class MemoryCompressionEncoder(nn.Module):
    """
    Memory Compression Encoder for Wan2.2-TI2V-5B.
    
    Compresses long video latents into short context features using a dual-branch
    architecture with optional UCPE camera encoding.
    
    HR branch: Convolutional downsampling compression
    LR branch: Pixel-space downsampling + VAE encoding + patchify (1,2,2)
    
    Args:
        in_channels: Number of input channels (VAE latent channels, 48 for Wan2.2, 16 for Wan2.1)
        out_dim: Output dimension (should match DiT hidden dimension, 2048 for TI2V-5B)
        compression_rate: Compression rate in format "TxHxW" (e.g., "2x4x4")
        hidden_channels: List of hidden channel dimensions
        num_heads: Number of attention heads
        image_height: Original image height (pixel space)
        image_width: Original image width (pixel space)
        mem_enc_use_ucpe: Whether to use UCPE camera encoding in Memory Encoder
        use_lr_branch: Whether to use low-resolution branch
        vae: Optional VAE model for LR branch encoding (not registered as submodule)
    """
    
    def __init__(
        self,
        in_channels: int = 48,  # VAE latent channels (48 for Wan2.2, 16 for Wan2.1)
        out_dim: int = 2048,  # DiT hidden dimension for Wan2.2-TI2V-5B
        compression_rate: str = "2x4x4",  # t2h4w4 compression ratio (TxHxW)
        hidden_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        num_heads: int = 8,
        image_height: int = 480,
        image_width: int = 832,
        mem_enc_use_ucpe: bool = True,
        use_lr_branch: bool = True,
        vae: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.compression_rate = compression_rate
        self.mem_enc_use_ucpe = mem_enc_use_ucpe
        self.use_lr_branch = use_lr_branch
        self.image_height = image_height
        self.image_width = image_width
        
        # Store VAE reference for LR branch (not registered as submodule to avoid training)
        object.__setattr__(self, '_vae', vae)
        
        # Parse compression rate (TxHxW)
        parts = compression_rate.lower().split('x')
        self.compress_t = int(parts[0])
        self.compress_h = int(parts[1])
        self.compress_w = int(parts[2])
        
        # VAE compression ratios
        self.vae_spatial_compress = 8   # VAE spatial compression
        self.vae_temporal_compress = 4  # VAE temporal compression
        
        # DiT patchify compression (1, 2, 2) - no temporal, 2x2 spatial
        self.patchify_t = 1
        self.patchify_h = 2
        self.patchify_w = 2
        
        # Calculate LR pixel downsample ratio
        # Target: LR branch patchify output size = HR branch output size
        # pixel_downsample = compression_rate / patchify
        self.lr_pixel_downsample_h = self.compress_h // self.patchify_h  # 4/2=2
        self.lr_pixel_downsample_w = self.compress_w // self.patchify_w  # 4/2=2
        self.lr_pixel_downsample_t = self.compress_t // self.patchify_t  # 2/1=2
        
        # DiT patch sizes (for P_inv encoding space alignment)
        # Consistent with ucpe_camera_control.py patch_dit: patch_factor = vae_spatial_compress * 2
        dit_patch_factor = self.vae_spatial_compress * 2  # 8 * 2 = 16
        self.dit_patches_x = image_width // dit_patch_factor   # 832/16 = 52
        self.dit_patches_y = image_height // dit_patch_factor   # 480/16 = 30
        
        # Memory Encoder compressed patch sizes (for self-attention etc.)
        # Compute based on scaled latent dimensions to handle non-divisible cases
        vae_h = image_height // self.vae_spatial_compress  # e.g., 480/8 = 60
        vae_w = image_width // self.vae_spatial_compress   # e.g., 832/8 = 104
        scaled_vae_h = math.ceil(vae_h / self.compress_h) * self.compress_h  # e.g., ceil(60/8)*8 = 64
        scaled_vae_w = math.ceil(vae_w / self.compress_w) * self.compress_w  # e.g., ceil(104/8)*8 = 104
        self.patches_x = scaled_vae_w // self.compress_w  # e.g., 104/8 = 13
        self.patches_y = scaled_vae_h // self.compress_h  # e.g., 64/8 = 8
        
        # High-resolution branch
        self.hr_blocks = nn.ModuleList()
        current_channels = in_channels
        
        # Compute downsampling strides for each layer
        hr_strides = self._compute_hr_strides(self.compress_h, self.compress_w, self.compress_t, len(hidden_channels))
        
        # Build HR branch blocks
        for i, (out_ch, stride) in enumerate(zip(hidden_channels, hr_strides)):
            self.hr_blocks.append(DownsampleBlock3D(current_channels, out_ch, stride=stride))
            # Add extra ResNet block for 256+ channel layers
            if out_ch >= 256:
                self.hr_blocks.append(ResnetBlock3D(out_ch, out_ch))
            current_channels = out_ch
        
        # Low-resolution branch
        if use_lr_branch:
            # LR patchify: Conv3d with kernel_size=(1,2,2), stride=(1,2,2)
            self.lr_patchify = nn.Conv3d(
                in_channels,
                hidden_channels[-1],
                kernel_size=(self.patchify_t, self.patchify_h, self.patchify_w),
                stride=(self.patchify_t, self.patchify_h, self.patchify_w)
            )
            # LR branch feature extraction
            self.lr_resnet = ResnetBlock3D(hidden_channels[-1], hidden_channels[-1])
            # LR branch projection to out_dim
            self.lr_proj_out = nn.Linear(hidden_channels[-1], out_dim)
            # Initialize LR output projection to zero for stable training
            nn.init.zeros_(self.lr_proj_out.weight)
            nn.init.zeros_(self.lr_proj_out.bias)
        
        # Self-Attention layer between hr_blocks and conv_out
        # Note: UCPE disabled here, as intermediate self-attention doesn't need it
        # UCPE is mainly used for the final output P_inv transform
        self.self_attn = SelfAttention3D(
            dim=current_channels,
            num_heads=num_heads,
            patches_x=max(1, self.patches_x),
            patches_y=max(1, self.patches_y),
            image_width=image_width,
            image_height=image_height,
            mem_enc_use_ucpe=False,  # Disable UCPE to avoid size mismatch
        )
        
        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=_get_num_groups(current_channels), num_channels=current_channels, eps=1e-6)
        self.conv_out = nn.Conv3d(current_channels, current_channels, kernel_size=3, padding=1)
        self.proj_out = nn.Linear(current_channels, out_dim)
        
        # Initialize output projection to zero for stable training
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
        # PRoPE transform
        # P_inv head config must match DiT-side PRoPE
        # TI2V-5B: num_heads=16, head_dim=128 (dim=2048)
        self.prope_num_heads = out_dim // 128  # 3072 // 128 = 24 for TI2V-5B
        self.prope_head_dim = 128
        
        if mem_enc_use_ucpe:
            # Use DiT-side patches for initialization to ensure P_inv aligns with DiT P
            self.prope_transform = PropeDotProductAttention(
                head_dim=self.prope_head_dim,
                patches_x=self.dit_patches_x,
                patches_y=self.dit_patches_y,
                image_width=image_width,
                image_height=image_height,
                freq_base=100.0,
                freq_scale=1.0,
                precompute_coeffs=True,
            )
    
    def _compute_hr_strides(
        self,
        compress_h: int,
        compress_w: int,
        compress_t: int,
        num_blocks: int
    ) -> List[Tuple[int, int, int]]:
        """
        Dynamically compute HR branch strides for target compression rate.
        
        Each dimension's compression must be a power of 2. Strides are distributed
        across blocks, with earlier active blocks getting all-dimension strides
        and later blocks getting spatial-only strides.
        
        Examples:
          2x4x4 with 4 blocks -> [(1,1,1), (1,1,1), (2,2,2), (1,2,2)]
          4x8x8 with 4 blocks -> [(1,1,1), (2,2,2), (2,2,2), (1,2,2)]
          8x8x8 with 4 blocks -> [(1,1,1), (2,2,2), (2,2,2), (2,2,2)]
        """
        n_t = int(math.log2(compress_t)) if compress_t > 1 else 0
        n_h = int(math.log2(compress_h)) if compress_h > 1 else 0
        n_w = int(math.log2(compress_w)) if compress_w > 1 else 0
        
        # Verify compression rates are powers of 2
        assert compress_t == 2 ** n_t, f"compress_t={compress_t} must be a power of 2"
        assert compress_h == 2 ** n_h, f"compress_h={compress_h} must be a power of 2"
        assert compress_w == 2 ** n_w, f"compress_w={compress_w} must be a power of 2"
        
        max_steps = max(n_h, n_w, n_t)
        assert max_steps <= num_blocks, (
            f"Need at least {max_steps} blocks for {compress_h}x{compress_w}x{compress_t} "
            f"compression, but only {num_blocks} blocks available"
        )
        
        strides = [(1, 1, 1)] * num_blocks
        
        # Assign downsampling strides to the last max_steps blocks
        start_idx = num_blocks - max_steps
        for i in range(max_steps):
            block_idx = start_idx + i
            st = 2 if i < n_t else 1
            sh = 2 if i < n_h else 1
            sw = 2 if i < n_w else 1
            strides[block_idx] = (st, sh, sw)
        
        logger.info(f"HR branch strides for {compress_t}x{compress_h}x{compress_w}: {strides}")
        return strides
    
    def _downsample_pixel_video(
        self,
        pixel_video: torch.Tensor,
    ) -> torch.Tensor:
        """
        Downsample pixel video for LR branch.
        
        Args:
            pixel_video: [B, 3, T_pixel, H_pixel, W_pixel] original pixel video
        
        Returns:
            Downsampled pixel video [B, 3, T_lr, H_lr, W_lr]
        """
        B, C, T, H, W = pixel_video.shape
        
        target_t = max(1, T // self.lr_pixel_downsample_t)
        target_h = max(1, H // self.lr_pixel_downsample_h)
        target_w = max(1, W // self.lr_pixel_downsample_w)
        
        downsampled = F.interpolate(
            pixel_video,
            size=(target_t, target_h, target_w),
            mode='trilinear',
            align_corners=False
        )
        
        return downsampled
    
    def forward(
        self,
        x: torch.Tensor,
        pixel_video: Optional[torch.Tensor] = None,
        camera_params: Optional[Dict] = None,
        lr_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compress video latents into context features with optional P_inv camera encoding.
        
        HR branch: Convolutional downsampling of latent
        LR branch: Pixel-space downsampling + VAE encoding + patchify
                    OR directly use precomputed LR latent (when lr_latent is provided)
        
        Args:
            x: VAE latent tensor [B, C, T, H, W] (HR latent)
            pixel_video: Optional pixel video [B, 3, T_pixel, H_pixel, W_pixel] for LR branch
            camera_params: dict with camera parameters for UCPE encoding
                Relray mode (recommended):
                - 'c2w': [B, T_pixel, 4, 4] camera-to-world matrices (pixel-space temporal)
                - 'x_fov': [B] horizontal field of view in degrees
                - 'xi': [B] UCM mirror parameter
            lr_latent: Optional precomputed LR latent [B, C, T, H, W] for LR branch.
                       When provided, skips pixel_video downsampling and VAE encoding.
        
        Returns:
            Compressed context features [B, N, out_dim] where N is the compressed sequence length
        """
        B, C, T, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Auto-move module weights to input device if needed (e.g. when model is on CPU)
        first_param = next(self.parameters(), None)
        if first_param is not None and first_param.device.type != device.type:
            self.to(device=device)

        # Camera pose temporal check
        if camera_params is not None and 'c2w' in camera_params:
            c2w_input = camera_params['c2w']
            T_c2w = c2w_input.shape[1]
            T_pixel = T * 4 - 3  # latent -> pixel: T_pixel = (T_latent - 1) * 4 + 1
            
            assert T_c2w == T_pixel, (
                f"Camera pose temporal dimension mismatch! "
                f"Expected pixel-space poses with {T_pixel} frames, but got {T_c2w} frames. "
                f"Please pass pixel-space poses to Memory Encoder, "
                f"the internal downsampling will be handled automatically."
            )
        
        # Spatial scaling and temporal padding
        # Scale spatial dimensions to be divisible by compression rate
        need_spatial_scale = (H % self.compress_h != 0) or (W % self.compress_w != 0)
        if need_spatial_scale:
            scaled_h = math.ceil(H / self.compress_h) * self.compress_h
            scaled_w = math.ceil(W / self.compress_w) * self.compress_w
            x = F.interpolate(x, size=(T, scaled_h, scaled_w), mode='trilinear', align_corners=False)
            logger.debug(f"Memory Encoder spatial scaling: ({H}, {W}) -> ({scaled_h}, {scaled_w})")
        
        # Pad temporal dimension to be divisible by compression rate
        need_temporal_pad = T % self.compress_t != 0
        if need_temporal_pad:
            pad_t = self.compress_t - (T % self.compress_t)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t), mode='replicate')
            logger.debug(f"Memory Encoder temporal padding: T={T} -> T={x.shape[2]} (padded {pad_t})")
        
        # Update shape after scaling/padding
        B, C, T, H, W = x.shape
        
        # High-resolution branch
        hr = x
        for block in self.hr_blocks:
            hr = block(hr)
        
        # HR self-attention
        B_hr, C_hr, T_hr, H_hr, W_hr = hr.shape
        hr_flat = rearrange(hr, 'b c t h w -> b (t h w) c')
        
        # Process camera_params for HR branch temporal dimension
        camera_params_hr = None
        if camera_params is not None:
            if 'c2w' in camera_params:
                c2w = camera_params['c2w']
                x_fov = camera_params['x_fov']
                xi = camera_params['xi']
                
                # Two-step downsampling:
                # Step 1: VAE-style downsampling (pixel -> latent space)
                c2w_latent = vae_style_temporal_downsample(c2w, dim=1)
                
                # Step 2: Additional 2x compression (stride-2 uniform, matching HR Conv3d stride=2)
                if c2w_latent.shape[1] != T_hr:
                    indices = torch.linspace(0, c2w_latent.shape[1] - 1, T_hr, device=device).long()
                    c2w_hr = c2w_latent[:, indices]
                else:
                    c2w_hr = c2w_latent
                
                # Compute relray viewmats for HR branch
                viewmats_hr = compute_relray_viewmats(
                    c2w=c2w_hr,
                    x_fov=x_fov,
                    xi=xi,
                    patches_y=H_hr,
                    patches_x=W_hr,
                    device=device,
                    dtype=dtype,
                )
                camera_params_hr = {'viewmats': viewmats_hr}
        
        hr_flat = self.self_attn(
            hr_flat,
            camera_params=camera_params_hr,
            spatial_shape=(T_hr, H_hr, W_hr)
        )
        hr = rearrange(hr_flat, 'b (t h w) c -> b c t h w', t=T_hr, h=H_hr, w=W_hr)
        
        # Low-resolution branch
        lr_branch_active = False
        if self.use_lr_branch and lr_latent is not None:
            # Use precomputed LR latent directly (skip pixel downsample + VAE encode)
            lr_branch_active = True
            lr = self.lr_patchify(lr_latent.to(dtype))
            lr = self.lr_resnet(lr)
            
            # Ensure LR and HR spatial sizes match
            if lr.shape[2:] != hr.shape[2:]:
                lr = F.interpolate(lr, size=hr.shape[2:], mode='trilinear', align_corners=False)
            
            # LR branch projection to out_dim
            _, _, T_lr, H_lr, W_lr = lr.shape
            lr_flat = rearrange(lr, 'b c t h w -> b (t h w) c')
            lr_out = self.lr_proj_out(lr_flat)  # [B, N, out_dim]
        elif self.use_lr_branch and pixel_video is not None:
            # Original path: pixel-space downsampling + VAE encoding
            lr_branch_active = True
            # Step 1: Pixel-space downsampling
            lr_pixel = self._downsample_pixel_video(pixel_video)
            
            # Step 2: VAE encoding
            if self._vae is not None:
                with torch.no_grad():
                    # Convert [B, C, T, H, W] to list of [C, T, H, W]
                    lr_pixel_list = [lr_pixel[i] for i in range(lr_pixel.shape[0])]
                    lr_latent_enc = self._vae.encode(lr_pixel_list)
                    # Output is list of [C, T, H, W], stack to [B, C, T, H, W]
                    lr_latent_enc = torch.stack(lr_latent_enc, dim=0)
            else:
                lr_latent_enc = pixel_video
            
            # Step 3: Patchify (1, 2, 2) downsampling + feature extraction
            lr = self.lr_patchify(lr_latent_enc)
            lr = self.lr_resnet(lr)
            
            # Ensure LR and HR spatial sizes match
            if lr.shape[2:] != hr.shape[2:]:
                lr = F.interpolate(lr, size=hr.shape[2:], mode='trilinear', align_corners=False)
            
            # LR branch projection to out_dim
            _, _, T_lr, H_lr, W_lr = lr.shape
            lr_flat = rearrange(lr, 'b c t h w -> b (t h w) c')
            lr_out = self.lr_proj_out(lr_flat)  # [B, N, out_dim]
        
        # HR output projection
        B_hr, C_hr, T_hr, H_hr, W_hr = hr.shape
        hr_norm = self.norm_out(hr)
        hr_norm = F.silu(hr_norm)
        hr_conv = self.conv_out(hr_norm)
        
        T_out, H_out, W_out = T_hr, H_hr, W_hr
        hr_flat = rearrange(hr_conv, 'b c t h w -> b (t h w) c')
        hr_out = self.proj_out(hr_flat)  # [B, N, out_dim]
        
        # Merge branches
        if lr_branch_active:
            out = hr_out + lr_out
        else:
            out = hr_out
        
        # Apply PRoPE transform
        if self.mem_enc_use_ucpe and camera_params is not None:
            B_out, N, D = out.shape
            
            if 'c2w' in camera_params:
                c2w = camera_params['c2w']
                x_fov = camera_params['x_fov']
                xi = camera_params['xi']
                
                # Two-step downsampling:
                # Step 1: VAE-style downsampling (pixel -> latent space)
                c2w_latent = vae_style_temporal_downsample(c2w, dim=1)
                
                # Step 2: Additional 2x compression (stride-2 uniform)
                if c2w_latent.shape[1] != T_out:
                    indices = torch.linspace(0, c2w_latent.shape[1] - 1, T_out, device=device).long()
                    c2w_compressed = c2w_latent[:, indices]
                else:
                    c2w_compressed = c2w_latent
                
                # Compute relray viewmats using DiT-side patches for encoding space alignment
                viewmats_full = compute_relray_viewmats(
                    c2w=c2w_compressed,
                    x_fov=x_fov,
                    xi=xi,
                    patches_y=self.dit_patches_y,
                    patches_x=self.dit_patches_x,
                    device=device,
                    dtype=dtype,
                )  # [B, T_out * dit_patches_y * dit_patches_x, 4, 4]
                
                # Spatial downsampling: use linspace to sample exactly H_out x W_out
                # positions from DiT grid, ensuring viewmats token count matches out
                viewmats_full = viewmats_full.view(
                    B_out, T_out, self.dit_patches_y, self.dit_patches_x, 4, 4
                )
                h_indices = torch.linspace(0, self.dit_patches_y - 1, H_out, device=device).long()
                w_indices = torch.linspace(0, self.dit_patches_x - 1, W_out, device=device).long()
                viewmats_compressed = viewmats_full[:, :, h_indices][:, :, :, w_indices]
                viewmats_compressed = viewmats_compressed.reshape(
                    B_out, T_out * H_out * W_out, 4, 4
                )
                
                K_compressed = None
            else:
                # No valid camera params (c2w not found), skip P_inv transform
                grid_sizes = torch.tensor(
                    [[T_out, H_out, W_out]] * B, device=device, dtype=torch.long
                )
                return out, grid_sizes
            
            # Compute coeffs from DiT-side positions, downsampled to match compressed tokens
            dit_x_indices = torch.arange(self.dit_patches_x, device=device, dtype=torch.float32)
            dit_y_indices = torch.arange(self.dit_patches_y, device=device, dtype=torch.float32)
            sampled_x = dit_x_indices[w_indices].long()
            sampled_y = dit_y_indices[h_indices].long()
            
            coeffs_x = _rope_precompute_coeffs(
                torch.tile(sampled_x, (len(sampled_y),)),
                freq_base=100.0,
                freq_scale=1.0,
                feat_dim=self.prope_head_dim // 4,
                dtype=viewmats_compressed.dtype,
            )
            coeffs_y = _rope_precompute_coeffs(
                torch.repeat_interleave(sampled_y, len(sampled_x)),
                freq_base=100.0,
                freq_scale=1.0,
                feat_dim=self.prope_head_dim // 4,
                dtype=viewmats_compressed.dtype,
            )
            
            # Update prope_transform patches to match compressed spatial resolution
            self.prope_transform.patches_x = W_out
            self.prope_transform.patches_y = H_out
            
            # Precompute P_inv transform with new coeffs
            self.prope_transform._precompute_and_cache_apply_fns(
                viewmats=viewmats_compressed,
                Ks=K_compressed,
                coeffs_x=coeffs_x,
                coeffs_y=coeffs_y,
            )
            
            # Apply P_inv transform
            out = out.view(B_out, N, self.prope_num_heads, self.prope_head_dim).transpose(1, 2)
            out = self.prope_transform._apply_to_kv(out)
            out = out.transpose(1, 2).reshape(B_out, N, D)
        
        # Return compressed features and grid sizes for RoPE in DiT
        grid_sizes = torch.tensor(
            [[T_out, H_out, W_out]] * B, device=device, dtype=torch.long
        )  # [B, 3]
        return out, grid_sizes
    
    def get_compressed_shape(self, input_t: int, input_h: int, input_w: int) -> Tuple[int, int, int]:
        """
        Calculate the output shape after compression.
        Accounts for spatial scaling and temporal padding when dimensions
        are not divisible by the compression rate.
        
        Args:
            input_t: input temporal dimension (in latent space)
            input_h: input height dimension (in latent space)
            input_w: input width dimension (in latent space)
        
        Returns:
            (t_out, h_out, w_out): output dimensions after compression
        """
        scaled_h = math.ceil(input_h / self.compress_h) * self.compress_h
        scaled_w = math.ceil(input_w / self.compress_w) * self.compress_w
        padded_t = math.ceil(input_t / self.compress_t) * self.compress_t
        
        t_out = max(1, padded_t // self.compress_t)
        h_out = max(1, scaled_h // self.compress_h)
        w_out = max(1, scaled_w // self.compress_w)
        return t_out, h_out, w_out
    
    def get_compressed_length(self, input_length: int, input_height: int, input_width: int) -> int:
        """Calculate the output sequence length after compression."""
        t_out, h_out, w_out = self.get_compressed_shape(input_length, input_height, input_width)
        return t_out * h_out * w_out
    
    def print_compression_info(self, input_t: int, input_h: int, input_w: int):
        """
        Print compression ratio information.
        
        Args:
            input_t: input temporal dimension (in latent space)
            input_h: input height dimension (in latent space)
            input_w: input width dimension (in latent space)
        """
        mem_t, mem_h, mem_w = self.get_compressed_shape(input_t, input_h, input_w)
        
        # Original VAE latent tokens (after DiT patchify)
        dit_t = input_t // self.patchify_t
        dit_h = input_h // self.patchify_h
        dit_w = input_w // self.patchify_w
        original_tokens = dit_t * dit_h * dit_w
        
        # Memory tokens
        mem_tokens = mem_t * mem_h * mem_w
        
        ratio = original_tokens / mem_tokens if mem_tokens > 0 else float('inf')
        
        # Scaling/padding info
        scaled_h = math.ceil(input_h / self.compress_h) * self.compress_h
        scaled_w = math.ceil(input_w / self.compress_w) * self.compress_w
        padded_t = math.ceil(input_t / self.compress_t) * self.compress_t
        scale_info = ""
        if scaled_h != input_h or scaled_w != input_w:
            scale_info += f"\n  Spatial scaling: ({input_h}, {input_w}) -> ({scaled_h}, {scaled_w})"
        if padded_t != input_t:
            scale_info += f"\n  Temporal padding: T={input_t} -> T={padded_t}"
        
        logger.info(
            f"Memory Compression Info:\n"
            f"  Input latent shape: ({input_t}, {input_h}, {input_w})\n"
            f"  Compression rate: {self.compression_rate} (TxHxW)\n"
            f"  DiT tokens (after patchify): {dit_t}x{dit_h}x{dit_w} = {original_tokens}\n"
            f"  Memory tokens (after compression): {mem_t}x{mem_h}x{mem_w} = {mem_tokens}\n"
            f"  Compression ratio: {ratio:.1f}x\n"
            f"  Memory Encoder out_dim: {self.out_dim}"
            f"{scale_info}"
        )
        
        return {
            'original_tokens': original_tokens,
            'memory_tokens': mem_tokens,
            'compression_ratio': ratio,
            'mem_grid': (mem_t, mem_h, mem_w),
            'dit_grid': (dit_t, dit_h, dit_w),
        }
