"""
CaR: Unified Inference Script for Camera/Action/HardCut/Continue modes.

Built on Wan2.2-TI2V-5B with UCPE camera control + Memory Compression Encoder.

Four modes (--mode):
  camera   : I2V with explicit camera pose sequence (one image -> one video)
  action   : I2V with WASD action commands (autoregressive segments)
  hardcut  : I2V with WASD commands + 'skip:' prefix for hard-cut transitions
  continue : V2V continuation from a context video / image sequence

Coordinate convention (CV): x=right, y=down, z=forward.
Pose tensors are 3x4 c2w matrices stored relative to a chosen reference frame
(usually the first frame of the current segment).
"""

import argparse
import gc
import json
import logging
import os
import sys

import imageio
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from wan.modules.custom_model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.ucpe_camera_control import patch_dit, prepare_camera_input
from wan.modules.memory_encoder import (
    MemoryCompressionEncoder,
    vae_style_temporal_downsample,
    compute_relray_viewmats,
    scale_pixels_for_memory_compression,
)
from wan.configs import WAN_CONFIGS
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.utils import save_video, masks_like
from wan.utils.adaptive_shift import compute_adaptive_shift
from core.utils import guess_load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='[\033[34m%(asctime)s\033[0m] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def rotation_matrix_y(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_matrix_x(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def build_segment_poses_relative(instruction, frame_num, step_size,
                                  rotate_angle, pitch_angle=20.0):
    """Build relative c2w poses (first frame = identity) for one WASD instruction.

    Supports composite commands joined with '+', e.g. 'right+down' combines
    yaw and pitch in a single segment.
    """
    cam_x = np.array([1, 0, 0], dtype=np.float64)
    cam_z = np.array([0, 0, 1], dtype=np.float64)

    rotation_end = np.eye(3, dtype=np.float64)
    translation_end = np.zeros(3, dtype=np.float64)

    bases = [b.strip() for b in str(instruction).split('+') if b.strip()]
    if not bases:
        raise ValueError(f"Empty instruction: '{instruction}'")

    for base in bases:
        if base == 'w':
            translation_end = translation_end + cam_z * step_size
        elif base == 's':
            translation_end = translation_end - cam_z * step_size
        elif base == 'a':
            translation_end = translation_end - cam_x * step_size
        elif base == 'd':
            translation_end = translation_end + cam_x * step_size
        elif base == 'left':
            rotation_end = rotation_end @ rotation_matrix_y(-rotate_angle)
        elif base == 'right':
            rotation_end = rotation_end @ rotation_matrix_y(rotate_angle)
        elif base == 'down':
            rotation_end = rotation_end @ rotation_matrix_x(-pitch_angle)
        elif base == 'up':
            rotation_end = rotation_end @ rotation_matrix_x(pitch_angle)
        else:
            raise ValueError(
                f"Unknown WASD instruction component '{base}' in '{instruction}'. "
                f"Expected: w, s, a, d, left, right, up, down."
            )

    rot_start = Rotation.from_matrix(np.eye(3))
    rot_end = Rotation.from_matrix(rotation_end)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_start, rot_end]))

    rel_c2ws = []
    for i in range(frame_num):
        alpha = i / max(frame_num - 1, 1)
        rot = slerp(alpha).as_matrix()
        trans = alpha * translation_end
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot
        c2w[:3, 3] = trans
        rel_c2ws.append(c2w)
    return rel_c2ws


def relative_to_absolute(rel_c2ws, ref_c2w):
    ref = np.array(ref_c2w, dtype=np.float64)
    return [ref @ np.array(rel, dtype=np.float64) for rel in rel_c2ws]


def abs_to_relative_poses(c2ws, ref_c2w):
    """Return [N, 3, 4] torch tensor of relative poses w.r.t. ref_c2w."""
    ref_w2c = np.linalg.inv(ref_c2w)
    rels = []
    for c2w in c2ws:
        rel = ref_w2c @ c2w
        rels.append(torch.as_tensor(rel[:3, :], dtype=torch.float32))
    return torch.stack(rels, dim=0)


def rel_c2ws_to_pose_tensor(rel_c2ws):
    """list of [4,4] np -> [N, 3, 4] torch tensor."""
    return torch.stack(
        [torch.as_tensor(c[:3, :], dtype=torch.float32) for c in rel_c2ws],
        dim=0,
    )


def parse_instruction(raw_cmd):
    """Parse a single motion command, return (base_cmd, skip_render).

    'skip:w' / 'SKIP:left' / 'skip:right+down' all mark a teleport-only step.
    """
    cmd = raw_cmd.strip().lower()
    if cmd.startswith('skip:'):
        return cmd[len('skip:'):].strip(), True
    return cmd, False


def crop_and_resize(image, target_width=832, target_height=480):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = TF.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    image = TF.center_crop(image, (target_height, target_width))
    return image


def video_tensor_to_last_frame_pil(video_tensor):
    """[C, T, H, W] in [-1, 1] -> PIL Image (last frame)."""
    last = video_tensor[:, -1]
    return TF.to_pil_image((last + 1.0) / 2.0)


def load_image_as_pixel_tensor(image_path, target_width, target_height):
    """Load a single image as [C, 1, H, W] tensor in [-1, 1]."""
    img = Image.open(image_path).convert('RGB')
    img = crop_and_resize(img, target_width, target_height)
    t = TF.to_tensor(img) * 2.0 - 1.0  # [-1, 1]
    return t.unsqueeze(1)  # [C, 1, H, W]


def load_video_as_pixel_tensor(video_path, target_width, target_height):
    """Load all frames from video file as [C, T, H, W] tensor in [-1, 1]."""
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        pil = Image.fromarray(frame)
        pil = crop_and_resize(pil, target_width, target_height)
        frames.append(TF.to_tensor(pil))
    reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    t = torch.stack(frames, dim=1) * 2.0 - 1.0  # [C, T, H, W]
    return t


def load_image_sequence(folder_path, target_width, target_height):
    """Load all images in a folder (sorted) as [C, T, H, W] tensor in [-1, 1]."""
    files = []
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(os.path.join(folder_path, fname))
    if not files:
        raise FileNotFoundError(f"No image files in {folder_path}")
    frames = []
    for p in files:
        img = Image.open(p).convert('RGB')
        img = crop_and_resize(img, target_width, target_height)
        frames.append(TF.to_tensor(img))
    t = torch.stack(frames, dim=1) * 2.0 - 1.0
    return t


def load_input_as_image_sequence(input_path, target_width, target_height):
    """Unify input to image sequence [C, T, H, W] in [-1, 1].

    Auto-detects:
      - Single image file (.png/.jpg/...) -> [C, 1, H, W]
      - Single video file (.mp4/.avi/...) -> [C, T, H, W]
      - Folder of images -> [C, T, H, W]
    Returns (pixel_tensor, kind) where kind in {'image', 'video', 'folder'}.
    """
    if os.path.isdir(input_path):
        return load_image_sequence(input_path, target_width, target_height), 'folder'
    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTS:
        return load_video_as_pixel_tensor(input_path, target_width, target_height), 'video'
    if ext in IMAGE_EXTS:
        return load_image_as_pixel_tensor(input_path, target_width, target_height), 'image'
    raise ValueError(
        f"Unrecognized input '{input_path}'. Expected a video file ({VIDEO_EXTS}), "
        f"image file ({IMAGE_EXTS}), or folder of images."
    )


def load_poses(path):
    """Load pose sequence from .json or .npy file -> list of [4, 4] np arrays.

    JSON format: list of 3x4 or 4x4 matrices (each as nested list).
    NPY format: array of shape [N, 3, 4] or [N, 4, 4].
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'poses' in data:
            data = data['poses']
        arr = np.array(data, dtype=np.float64)
    elif ext in {'.npy', '.npz'}:
        arr = np.load(path)
        if ext == '.npz':
            arr = arr['poses'] if 'poses' in arr.files else arr[arr.files[0]]
        arr = np.asarray(arr, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported pose file extension: {ext}")

    if arr.ndim != 3 or arr.shape[1] not in (3, 4) or arr.shape[2] != 4:
        raise ValueError(
            f"Pose file {path} must contain [N, 3, 4] or [N, 4, 4]. Got {arr.shape}."
        )

    out = []
    for m in arr:
        if m.shape == (3, 4):
            full = np.eye(4, dtype=np.float64)
            full[:3, :4] = m
            out.append(full)
        else:
            out.append(m.astype(np.float64))
    return out


def save_poses(c2ws, path):
    """Save list of [4, 4] np arrays as JSON."""
    data = [c.tolist() for c in c2ws]
    with open(path, 'w') as f:
        json.dump({'poses': data}, f, indent=2)


def build_mem_camera_input(context_c2ws, ref_c2w, mem_grid, model,
                            x_fov, xi, device, dtype):
    if not context_c2ws or mem_grid is None:
        return None

    mem_f = int(mem_grid[0, 0].item())
    mem_h = int(mem_grid[0, 1].item())
    mem_w = int(mem_grid[0, 2].item())
    dit_x = model.patches_x
    dit_y = model.patches_y

    rel = abs_to_relative_poses(context_c2ws, ref_c2w)
    ctx = rel.unsqueeze(0).to(device=device, dtype=dtype)

    n = ctx.shape[1]
    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w = c2w.unsqueeze(0).unsqueeze(0).expand(1, n, -1, -1).clone()
    c2w[..., :3, :4] = ctx

    c2w_lat = vae_style_temporal_downsample(c2w, dim=1)
    if c2w_lat.shape[1] != mem_f:
        idx = torch.linspace(0, c2w_lat.shape[1] - 1, mem_f, device=device).long()
        c2w_lat = c2w_lat[:, idx]

    viewmats = compute_relray_viewmats(
        c2w=c2w_lat, x_fov=x_fov, xi=xi,
        patches_y=dit_y, patches_x=dit_x,
        device=device, dtype=dtype,
    )
    viewmats = viewmats.view(1, mem_f, dit_y, dit_x, 4, 4)
    h_idx = torch.linspace(0, dit_y - 1, mem_h, device=device).long()
    w_idx = torch.linspace(0, dit_x - 1, mem_w, device=device).long()
    viewmats = viewmats[:, :, h_idx][:, :, :, w_idx]
    viewmats = viewmats.reshape(1, mem_f * mem_h * mem_w, 4, 4)
    return {"viewmats": viewmats}


def find_closest_pose_frame(context_c2ws, ref_c2w):
    ref_w2c = np.linalg.inv(ref_c2w)
    eye34 = np.eye(4, dtype=np.float64)[:3, :]
    best_idx, min_d = 0, float('inf')
    for i, c in enumerate(context_c2ws):
        d = np.linalg.norm((ref_w2c @ c)[:3, :] - eye34, 'fro')
        if d < min_d:
            min_d, best_idx = d, i
    return best_idx, min_d


def encode_hr_frame(vae, model, frame_pixel, device, param_dtype):
    frame_input = frame_pixel.unsqueeze(1).to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        latent = vae.encode([frame_input])[0]
    latent_input = latent.unsqueeze(0)
    pe = next(model.patch_embedding.parameters())
    latent_input = latent_input.to(device=pe.device, dtype=pe.dtype)
    with torch.no_grad():
        tokens = model.patch_embedding(latent_input)
    grid = torch.tensor([list(tokens.shape[2:])], dtype=torch.long, device=device)
    tokens = tokens.flatten(2).transpose(1, 2).to(device=device, dtype=param_dtype)
    return tokens, grid


def build_hr_camera_input(hr_c2w, ref_c2w, hr_grid, model,
                           x_fov, xi, device, dtype):
    h = int(hr_grid[0, 1].item())
    w = int(hr_grid[0, 2].item())
    rel = abs_to_relative_poses([hr_c2w], ref_c2w)
    c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    c2w[..., :3, :4] = rel.unsqueeze(0).to(device=device, dtype=dtype)
    viewmats = compute_relray_viewmats(
        c2w=c2w, x_fov=x_fov, xi=xi,
        patches_y=h, patches_x=w,
        device=device, dtype=dtype,
    )
    return {"viewmats": viewmats}


@torch.no_grad()
def i2v_generate(
    model, text_encoder, vae, input_prompt, camera_input,
    img=None,
    height=480, width=832, frame_num=81,
    num_train_timesteps=1000, shift=5.0,
    sample_solver='unipc', sampling_steps=50,
    guide_scale=5.0, n_prompt="", seed=0,
    device=torch.device('cuda'), param_dtype=torch.bfloat16,
    memory_tokens=None, mem_grid_sizes=None,
    rope_mode='rope+memrope', mem_camera_input=None, mem_compress_t=1,
    hr_tokens=None, hr_grid_sizes=None, hr_camera_input=None, hr_t_pos=-4200.0,
    mem_cfg_drop=False, hr_cfg_drop=False,
    cfg_low=0.0, cfg_high=1.0,
    adaptive_shift=False, shift_min=3.5, shift_max=8.0,
):
    """One-shot generation. img=None -> T2V; img=PIL -> I2V (first frame clean)."""
    vae_stride = (4, 16, 16)
    patch_size = (1, 2, 2)
    F = frame_num
    seq_len = (
        ((F - 1) // vae_stride[0] + 1)
        * (height // vae_stride[1])
        * (width // vae_stride[2])
        // (patch_size[1] * patch_size[2])
    )

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    noise = torch.randn(
        vae.model.z_dim,
        (F - 1) // vae_stride[0] + 1,
        height // vae_stride[1],
        width // vae_stride[2],
        dtype=torch.float32, generator=seed_g, device=device,
    )

    text_encoder.model.to(device)
    context = text_encoder([input_prompt], device)
    context_null = text_encoder([n_prompt], device)
    text_encoder.model.cpu()
    torch.cuda.empty_cache()

    if img is not None:
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device).unsqueeze(1)
        z = vae.encode([img_tensor])
    else:
        z = None

    if adaptive_shift:
        latent_frames = (frame_num - 1) // 4 + 1
        shift = compute_adaptive_shift(height, width, latent_frames,
                                       shift_min=shift_min, shift_max=shift_max)

    with torch.amp.autocast('cuda', dtype=param_dtype):
        if sample_solver == 'unipc':
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=1, use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == 'dpm++':
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                shift=1, use_dynamic_shifting=False,
            )
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")

        if img is not None:
            _, mask2 = masks_like([noise], zero=True)
            latent = (1.0 - mask2[0]) * z[0] + mask2[0] * noise
        else:
            latent = noise
            mask2 = None

        arg_c = {'context': [context[0]], 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        model.to(device)
        torch.cuda.empty_cache()

        for t in tqdm(timesteps, desc="Denoising", disable=False):
            latent_model_input = [latent.to(device)]
            timestep = torch.stack([t]).to(device)
            if mask2 is not None:
                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep,
                ])
                timestep = temp_ts.unsqueeze(0)
            else:
                timestep = timestep.expand(1, seq_len)

            t_norm = float(t.item()) / float(num_train_timesteps)
            do_cfg = cfg_low <= t_norm <= cfg_high

            noise_pred_cond = model(
                latent_model_input, t=timestep,
                control_camera_input=camera_input,
                memory_tokens=memory_tokens,
                mem_grid_sizes=mem_grid_sizes,
                rope_mode=rope_mode,
                mem_camera_input=mem_camera_input,
                mem_compress_t=mem_compress_t,
                hr_tokens=hr_tokens,
                hr_grid_sizes=hr_grid_sizes,
                hr_camera_input=hr_camera_input,
                hr_t_pos=hr_t_pos,
                **arg_c
            )[0]

            if do_cfg:
                u_mem_t = None if mem_cfg_drop else memory_tokens
                u_mem_g = None if mem_cfg_drop else mem_grid_sizes
                u_mem_c = None if mem_cfg_drop else mem_camera_input
                u_hr_t = None if hr_cfg_drop else hr_tokens
                u_hr_g = None if hr_cfg_drop else hr_grid_sizes
                u_hr_c = None if hr_cfg_drop else hr_camera_input

                noise_pred_uncond = model(
                    latent_model_input, t=timestep,
                    control_camera_input=camera_input,
                    memory_tokens=u_mem_t, mem_grid_sizes=u_mem_g,
                    rope_mode=rope_mode,
                    mem_camera_input=u_mem_c, mem_compress_t=mem_compress_t,
                    hr_tokens=u_hr_t, hr_grid_sizes=u_hr_g,
                    hr_camera_input=u_hr_c, hr_t_pos=hr_t_pos,
                    **arg_null
                )[0]
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0), t,
                latent.unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)
            if mask2 is not None:
                latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

            del latent_model_input

        model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        videos = vae.decode([latent])

    del noise, latent, scheduler
    gc.collect()
    torch.cuda.synchronize()
    return videos[0]  # [C, T, H, W]


def load_models(args, device):
    config = WAN_CONFIGS['ti2v-5B']

    logger.info("Loading T5 text encoder ...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.checkpoint_dir, config.t5_tokenizer),
    )

    logger.info("Loading VAE ...")
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.checkpoint_dir, config.vae_checkpoint),
        device=device,
    )

    logger.info("Loading DiT model ...")
    model = WanModel.from_pretrained(args.checkpoint_dir, low_cpu_mem_usage=True)

    logger.info(f"Patching DiT with camera_condition={args.camera_condition} ...")
    patch_dit(
        model,
        method=args.camera_condition,
        height=args.height, width=args.width,
        vae_downscale_factor=args.vae_downscale,
        attn_compress=args.attn_compress,
        adaptation_method=args.adaptation_method,
        enable_camera=True,
    )

    if args.enable_memory:
        hidden = tuple(int(x) for x in args.memory_hidden_channels.split(','))
        mem_enc = MemoryCompressionEncoder(
            in_channels=args.memory_in_channels,
            out_dim=args.memory_out_dim,
            compression_rate=args.memory_compression_rate,
            hidden_channels=hidden,
            num_heads=args.memory_num_heads,
            image_height=args.height,
            image_width=args.width,
            mem_enc_use_ucpe=args.memory_use_ucpe,
            use_lr_branch=args.memory_use_lr_branch,
            vae=vae,
        )
        model.memory_encoder = mem_enc
        logger.info(f"Memory Encoder enabled (compression={args.memory_compression_rate}).")

    if args.car_checkpoint:
        logger.info(f"Loading checkpoint from {args.car_checkpoint} ...")
        sd = guess_load_checkpoint(args.car_checkpoint)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            logger.warning(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    model.to(config.param_dtype)
    if args.enable_memory and hasattr(model, 'memory_encoder'):
        model.memory_encoder.to(device=device, dtype=config.param_dtype)

    mem_compress_t = 1
    if args.enable_memory and hasattr(model, 'memory_encoder'):
        mem_compress_t = model.memory_encoder.compress_t

    return {
        'model': model,
        'vae': vae,
        'text_encoder': text_encoder,
        'config': config,
        'device': device,
        'mem_compress_t': mem_compress_t,
        'n_prompt': config.sample_neg_prompt,
    }


def encode_segment_conditioning(
    args, ctx, ref_c2w, segment_rel_c2ws, segment_abs_c2ws,
    context_pixel_segments, context_c2ws,
    x_fov_t, xi_t,
):
    """Build camera_input, mem_tokens, mem_camera_input, hr_tokens, hr_camera_input.

    Memory and HR are derived from the accumulated context (pixel + poses).
    """
    model = ctx['model']
    vae = ctx['vae']
    config = ctx['config']
    device = ctx['device']
    dtype = config.param_dtype

    # Camera (target trajectory)
    pose_tensor = rel_c2ws_to_pose_tensor(segment_rel_c2ws).unsqueeze(0).to(
        device=device, dtype=torch.bfloat16,
    )
    camera_input = prepare_camera_input(
        pose=pose_tensor, x_fov=x_fov_t, xi=xi_t,
        method=args.camera_condition,
        patches_x=model.patches_x, patches_y=model.patches_y,
        width=args.width, height=args.height,
        device=device, dtype=dtype,
    )

    mem_tokens = mem_grid = mem_camera_input = None
    if args.enable_memory and context_pixel_segments:
        ctx_pixel = torch.cat(context_pixel_segments, dim=1)
        ctx_video = ctx_pixel.to(device=device, dtype=vae.dtype)
        ctx_video = scale_pixels_for_memory_compression(
            ctx_video, args.memory_compression_rate)
        with torch.no_grad():
            ctx_latent = vae.encode([ctx_video])[0].unsqueeze(0).to(
                device=device, dtype=dtype)

        camera_params = None
        if args.memory_use_ucpe and context_c2ws:
            rel = abs_to_relative_poses(context_c2ws, ref_c2w)
            n = len(context_c2ws)
            c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(
                1, n, -1, -1).clone()
            c2w[..., :3, :4] = rel.unsqueeze(0).to(device=device, dtype=dtype)
            camera_params = {'c2w': c2w, 'x_fov': x_fov_t, 'xi': xi_t}

        with torch.no_grad():
            mem_tokens, mem_grid = model.memory_encoder(ctx_latent, camera_params=camera_params)
        mem_camera_input = build_mem_camera_input(
            context_c2ws, ref_c2w, mem_grid, model,
            x_fov_t, xi_t, device, dtype,
        )
        del ctx_video, ctx_latent
        torch.cuda.empty_cache()

    hr_tokens = hr_grid = hr_camera_input = None
    if args.enable_hr_frame and context_c2ws:
        if args.hr_frame_mode == 'first':
            hr_idx = 0
        else:
            hr_idx, _ = find_closest_pose_frame(context_c2ws, segment_abs_c2ws[-1])
        ctx_pixel = torch.cat(context_pixel_segments, dim=1)
        hr_pixel = ctx_pixel[:, hr_idx]
        hr_tokens, hr_grid = encode_hr_frame(vae, model, hr_pixel, device, dtype)
        hr_camera_input = build_hr_camera_input(
            context_c2ws[hr_idx], ref_c2w, hr_grid, model,
            x_fov_t, xi_t, device, dtype,
        )
        del ctx_pixel
        torch.cuda.empty_cache()

    return {
        'camera_input': camera_input,
        'mem_tokens': mem_tokens, 'mem_grid': mem_grid,
        'mem_camera_input': mem_camera_input,
        'hr_tokens': hr_tokens, 'hr_grid': hr_grid,
        'hr_camera_input': hr_camera_input,
    }


def prepare_context(args, target_h, target_w):
    """Load input as image sequence; return (pixel_tensor [C,T,H,W], kind, T)."""
    pixels, kind = load_input_as_image_sequence(args.input_path, target_w, target_h)
    n_frames = pixels.shape[1]
    logger.info(f"  Input loaded: kind={kind}, shape={tuple(pixels.shape)}, frames={n_frames}")
    return pixels, kind, n_frames


def validate_inputs(args):
    print("=" * 70)
    print(f"[CaR] Mode: {args.mode}")
    print(f"[CaR] Input: {args.input_path}")
    print(f"[CaR] Output: {args.output_dir}")

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"--input_path does not exist: {args.input_path}")

    # Probe input kind for messaging
    if os.path.isdir(args.input_path):
        n_frames = sum(
            1 for f in os.listdir(args.input_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        kind = 'folder'
    elif os.path.splitext(args.input_path)[1].lower() in VIDEO_EXTS:
        reader = imageio.get_reader(args.input_path)
        try:
            n_frames = reader.count_frames()
        except Exception:
            n_frames = sum(1 for _ in reader)
        reader.close()
        kind = 'video'
    elif os.path.splitext(args.input_path)[1].lower() in IMAGE_EXTS:
        n_frames = 1
        kind = 'image'
    else:
        raise ValueError(f"Unrecognized input format: {args.input_path}")

    print(f"[CaR] Input kind: {kind}, frames: {n_frames}")

    if args.mode == 'camera':
        print("[CaR] camera mode: I2V with explicit camera pose sequence.")
        if n_frames != 1:
            raise AssertionError(
                f"camera mode expects a single image input (got {n_frames} frames). "
                f"Use --mode continue for video input."
            )
        if not args.target_poses:
            raise AssertionError("camera mode requires --target_poses <file>.")
        poses = load_poses(args.target_poses)
        n_poses = len(poses)
        # Allow multi-segment trajectories: poses = frame_num + k * (frame_num - 1).
        # Each extra segment overlaps the previous by 1 frame and contributes
        # frame_num - 1 new poses.
        step = args.frame_num - 1
        if n_poses < args.frame_num or (n_poses - args.frame_num) % step != 0:
            raise AssertionError(
                f"target_poses has {n_poses} entries, but it must equal "
                f"frame_num + k*(frame_num-1) = {args.frame_num} + k*{step} "
                f"for some integer k >= 0 (e.g. {args.frame_num}, "
                f"{args.frame_num + step}, {args.frame_num + 2*step}, ...)."
            )
        n_segments = 1 + (n_poses - args.frame_num) // step
        print(f"[CaR] Loaded {n_poses} target poses -> {n_segments} segment(s) of "
              f"{args.frame_num} frames (overlap 1). OK.")

    elif args.mode == 'action':
        print("[CaR] action mode: I2V with WASD action commands "
              "(autoregressive segments, no skip).")
        if n_frames != 1:
            raise AssertionError(
                f"action mode expects a single image input (got {n_frames} frames)."
            )
        if not args.motion_sequence:
            raise AssertionError("action mode requires --motion_sequence \"w,a,d,...\".")
        if 'skip:' in args.motion_sequence.lower():
            raise AssertionError(
                "action mode does not allow 'skip:' commands. Use --mode hardcut for hard cuts."
            )
        cmds = [c for c in args.motion_sequence.split(',') if c.strip()]
        print(f"[CaR] Parsed {len(cmds)} WASD commands: {cmds}")

    elif args.mode == 'hardcut':
        print("[CaR] hardcut mode: I2V with WASD + 'skip:' for hard-cut transitions.")
        if n_frames != 1:
            raise AssertionError(
                f"hardcut mode expects a single image input (got {n_frames} frames)."
            )
        if not args.motion_sequence:
            raise AssertionError("hardcut mode requires --motion_sequence with 'skip:' segments.")
        if 'skip:' not in args.motion_sequence.lower():
            print("[CaR] WARNING: hardcut mode without any 'skip:' command behaves like action mode.")
        cmds = [c for c in args.motion_sequence.split(',') if c.strip()]
        print(f"[CaR] Parsed {len(cmds)} commands: {cmds}")

    elif args.mode == 'continue':
        print("[CaR] continue mode: V2V continuation from context video + WASD action commands.")
        if n_frames < 2:
            raise AssertionError("continue mode requires at least 2 context frames.")
        if not args.context_poses:
            raise AssertionError(
                "continue mode requires --context_poses <file> matching the input frames. "
                f"Input has {n_frames} frame(s)."
            )
        ctx_poses = load_poses(args.context_poses)
        if len(ctx_poses) != n_frames:
            raise AssertionError(
                f"context_poses has {len(ctx_poses)} entries, but input has {n_frames} frames. "
                f"They MUST match (one pose per frame)."
            )
        if not args.motion_sequence:
            raise AssertionError("continue mode requires --motion_sequence for the new segments.")
        cmds = [c for c in args.motion_sequence.split(',') if c.strip()]
        print(f"[CaR] Context: {n_frames} frames + {len(ctx_poses)} poses. "
              f"Action commands: {len(cmds)}. Parsed: {cmds}")

    else:
        raise ValueError(f"Unknown --mode: {args.mode}")
    print("=" * 70)


def run_camera_mode(args, ctx):
    """Single image + explicit pose sequence -> one (or many) videos.

    If ``len(target_poses) == frame_num`` the output is a single segment.
    Otherwise the long pose sequence is split into overlap-by-1 segments of
    ``frame_num`` poses each, and segments are generated autoregressively
    (each segment's first frame is the previous segment's last frame).
    """
    device = ctx['device']
    config = ctx['config']
    dtype = config.param_dtype

    pixels, _, _ = prepare_context(args, args.height, args.width)
    prompt = read_prompt(args)
    save_context_and_prompt(args, ctx, pixels, prompt)

    # Load full pose sequence and split into segments of frame_num poses
    # with overlap-by-1. ``target_poses[0]`` is treated as identity reference.
    target_poses_abs = load_poses(args.target_poses)
    ref0 = target_poses_abs[0]
    inv_ref0 = np.linalg.inv(ref0)
    target_global = [inv_ref0 @ p for p in target_poses_abs]  # first = identity

    step = args.frame_num - 1
    n_segments = 1 + (len(target_global) - args.frame_num) // step
    segments_global = []
    for s in range(n_segments):
        start = s * step
        end = start + args.frame_num
        segments_global.append(target_global[start:end])

    x_fov_t = torch.tensor([args.x_fov], device=device)
    xi_t = torch.tensor([args.xi], device=device)

    init_c2w = np.eye(4, dtype=np.float64)
    current_c2w = init_c2w.copy()
    current_first_frame = video_tensor_to_last_frame_pil(pixels)

    context_pixel_segments = [pixels]
    context_c2ws = [init_c2w]

    all_videos = []
    for seg_idx, seg_abs_global in enumerate(segments_global):
        logger.info(f"\n{'='*60}\nSegment {seg_idx+1}/{n_segments} (camera, "
                    f"poses [{seg_idx*step}..{seg_idx*step + args.frame_num}))")

        ref_c2w = current_c2w.copy()
        # Re-express segment's poses relative to its own first frame (=identity).
        inv_first = np.linalg.inv(seg_abs_global[0])
        seg_rel = [inv_first @ p for p in seg_abs_global]
        # Then map back to absolute world using ref_c2w as the anchor.
        seg_abs_world = relative_to_absolute(seg_rel, ref_c2w)

        cond = encode_segment_conditioning(
            args, ctx, ref_c2w=ref_c2w,
            segment_rel_c2ws=seg_rel,
            segment_abs_c2ws=seg_abs_world,
            context_pixel_segments=context_pixel_segments,
            context_c2ws=context_c2ws,
            x_fov_t=x_fov_t, xi_t=xi_t,
        )

        logger.info(f"  Generating {args.frame_num} frames (I2V camera) ...")
        video = i2v_generate(
            model=ctx['model'], text_encoder=ctx['text_encoder'], vae=ctx['vae'],
            input_prompt=prompt, camera_input=cond['camera_input'],
            img=current_first_frame,
            height=args.height, width=args.width, frame_num=args.frame_num,
            num_train_timesteps=config.num_train_timesteps,
            shift=args.shift, sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps, guide_scale=args.guide_scale,
            n_prompt=ctx['n_prompt'], seed=args.seed + seg_idx,
            device=device, param_dtype=dtype,
            memory_tokens=cond['mem_tokens'], mem_grid_sizes=cond['mem_grid'],
            rope_mode=args.rope_mode,
            mem_camera_input=cond['mem_camera_input'],
            mem_compress_t=ctx['mem_compress_t'],
            hr_tokens=cond['hr_tokens'], hr_grid_sizes=cond['hr_grid'],
            hr_camera_input=cond['hr_camera_input'],
            hr_t_pos=args.hr_t_pos,
            mem_cfg_drop=args.mem_cfg_drop, hr_cfg_drop=args.hr_cfg_drop,
            cfg_low=args.cfg_low, cfg_high=args.cfg_high,
            adaptive_shift=args.adaptive_shift,
            shift_min=args.shift_min, shift_max=args.shift_max,
        )
        all_videos.append(video.cpu())
        save_segment_incremental(args, ctx, all_videos, pixels, seg_idx)

        current_c2w = seg_abs_world[-1].copy()
        current_first_frame = video_tensor_to_last_frame_pil(video)
        # Accumulate generated frames into context (drop overlap frame).
        context_pixel_segments.append(video[:, 1:].cpu())
        context_c2ws.extend(seg_abs_world[1:])

        del video
        torch.cuda.empty_cache()
        gc.collect()


def run_action_or_hardcut_mode(args, ctx):
    """Single image + WASD commands; hardcut also processes 'skip:' segments."""
    device = ctx['device']
    config = ctx['config']
    dtype = config.param_dtype

    pixels, _, _ = prepare_context(args, args.height, args.width)
    prompt = read_prompt(args)
    save_context_and_prompt(args, ctx, pixels, prompt)

    raw_cmds = [c for c in args.motion_sequence.split(',') if c.strip()]
    instructions = [parse_instruction(c) for c in raw_cmds]
    if args.mode == 'action' and any(skip for _, skip in instructions):
        raise AssertionError("action mode received 'skip:' command (use --mode hardcut).")

    x_fov_t = torch.tensor([args.x_fov], device=device)
    xi_t = torch.tensor([args.xi], device=device)

    init_c2w = np.eye(4, dtype=np.float64)
    current_c2w = init_c2w.copy()
    current_first_frame = video_tensor_to_last_frame_pil(pixels)

    context_pixel_segments = [pixels]
    context_c2ws = [init_c2w]

    all_videos = []
    rendered_idx = 0
    for seg_idx, (instr, skip) in enumerate(instructions):
        tag = '[SKIP] ' if skip else ''
        logger.info(f"\n{'='*60}\n{tag}Segment {seg_idx+1}/{len(instructions)}: '{instr}'")

        ref_c2w = current_c2w.copy()
        rel = build_segment_poses_relative(
            instr, args.frame_num, args.step_size,
            args.rotate_angle, pitch_angle=args.pitch_angle,
        )
        abs_c2ws = relative_to_absolute(rel, ref_c2w)

        if skip:
            current_c2w = abs_c2ws[-1].copy()
            current_first_frame = None  # next segment starts in T2V mode
            logger.info(f"  [SKIP] camera advanced; next segment in T2V mode.")
            continue

        cond = encode_segment_conditioning(
            args, ctx, ref_c2w=ref_c2w,
            segment_rel_c2ws=rel, segment_abs_c2ws=abs_c2ws,
            context_pixel_segments=context_pixel_segments,
            context_c2ws=context_c2ws,
            x_fov_t=x_fov_t, xi_t=xi_t,
        )

        mode_str = 'I2V' if current_first_frame is not None else 'T2V'
        logger.info(f"  Generating {args.frame_num} frames ({mode_str}) ...")
        video = i2v_generate(
            model=ctx['model'], text_encoder=ctx['text_encoder'], vae=ctx['vae'],
            input_prompt=prompt, camera_input=cond['camera_input'],
            img=current_first_frame,
            height=args.height, width=args.width, frame_num=args.frame_num,
            num_train_timesteps=config.num_train_timesteps,
            shift=args.shift, sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps, guide_scale=args.guide_scale,
            n_prompt=ctx['n_prompt'], seed=args.seed + seg_idx,
            device=device, param_dtype=dtype,
            memory_tokens=cond['mem_tokens'], mem_grid_sizes=cond['mem_grid'],
            rope_mode=args.rope_mode,
            mem_camera_input=cond['mem_camera_input'],
            mem_compress_t=ctx['mem_compress_t'],
            hr_tokens=cond['hr_tokens'], hr_grid_sizes=cond['hr_grid'],
            hr_camera_input=cond['hr_camera_input'],
            hr_t_pos=args.hr_t_pos,
            mem_cfg_drop=args.mem_cfg_drop, hr_cfg_drop=args.hr_cfg_drop,
            cfg_low=args.cfg_low, cfg_high=args.cfg_high,
            adaptive_shift=args.adaptive_shift,
            shift_min=args.shift_min, shift_max=args.shift_max,
        )
        all_videos.append(video.cpu())
        seg_label = instr.replace("+", "-")
        save_segment_incremental(args, ctx, all_videos, pixels, seg_idx, seg_label=seg_label)
        rendered_idx += 1

        current_c2w = abs_c2ws[-1].copy()
        if current_first_frame is not None:
            current_first_frame = video_tensor_to_last_frame_pil(video)
        # Accumulate generated frames into context (skip first frame to avoid overlap)
        context_pixel_segments.append(video[:, 1:].cpu() if current_first_frame is not None else video.cpu())
        if current_first_frame is not None:
            context_c2ws.extend(abs_c2ws[1:])
        else:
            context_c2ws.extend(abs_c2ws)
        # After a skip segment, we lost continuity, so treat next as fresh I2V boundary
        if current_first_frame is None:
            current_first_frame = video_tensor_to_last_frame_pil(video)

        del video
        torch.cuda.empty_cache()
        gc.collect()


def run_continue_mode(args, ctx):
    """V2V continuation: multi-frame context video + WASD action commands.

    Like action mode but the context is an existing video (multiple frames)
    rather than a single image. Subsequent segments are generated
    autoregressively using the WASD motion_sequence commands.
    """
    device = ctx['device']
    config = ctx['config']
    dtype = config.param_dtype

    pixels, _, n_ctx = prepare_context(args, args.height, args.width)
    prompt = read_prompt(args)
    save_context_and_prompt(args, ctx, pixels, prompt)

    ctx_poses = load_poses(args.context_poses)

    # Normalize context poses: first pose becomes identity.
    ref = ctx_poses[0]
    inv_ref = np.linalg.inv(ref)
    ctx_c2ws_local = [inv_ref @ p for p in ctx_poses]

    # Start autoregressive generation from the last context pose.
    current_c2w = ctx_c2ws_local[-1].copy()
    current_first_frame = video_tensor_to_last_frame_pil(pixels)

    raw_cmds = [c for c in args.motion_sequence.split(',') if c.strip()]
    instructions = [parse_instruction(c) for c in raw_cmds]

    x_fov_t = torch.tensor([args.x_fov], device=device)
    xi_t = torch.tensor([args.xi], device=device)

    context_pixel_segments = [pixels]
    context_c2ws = list(ctx_c2ws_local)

    all_videos = []
    rendered_idx = 0
    for seg_idx, (instr, skip) in enumerate(instructions):
        tag = '[SKIP] ' if skip else ''
        logger.info(f"\n{'='*60}\n{tag}Segment {seg_idx+1}/{len(instructions)}: '{instr}'")

        ref_c2w = current_c2w.copy()
        rel = build_segment_poses_relative(
            instr, args.frame_num, args.step_size,
            args.rotate_angle, pitch_angle=args.pitch_angle,
        )
        abs_c2ws = relative_to_absolute(rel, ref_c2w)

        if skip:
            current_c2w = abs_c2ws[-1].copy()
            current_first_frame = None
            logger.info(f"  [SKIP] camera advanced; next segment in T2V mode.")
            continue

        cond = encode_segment_conditioning(
            args, ctx, ref_c2w=ref_c2w,
            segment_rel_c2ws=rel, segment_abs_c2ws=abs_c2ws,
            context_pixel_segments=context_pixel_segments,
            context_c2ws=context_c2ws,
            x_fov_t=x_fov_t, xi_t=xi_t,
        )

        mode_str = 'I2V' if current_first_frame is not None else 'T2V'
        logger.info(f"  Generating {args.frame_num} frames ({mode_str}, continue) ...")
        video = i2v_generate(
            model=ctx['model'], text_encoder=ctx['text_encoder'], vae=ctx['vae'],
            input_prompt=prompt, camera_input=cond['camera_input'],
            img=current_first_frame,
            height=args.height, width=args.width, frame_num=args.frame_num,
            num_train_timesteps=config.num_train_timesteps,
            shift=args.shift, sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps, guide_scale=args.guide_scale,
            n_prompt=ctx['n_prompt'], seed=args.seed + seg_idx,
            device=device, param_dtype=dtype,
            memory_tokens=cond['mem_tokens'], mem_grid_sizes=cond['mem_grid'],
            rope_mode=args.rope_mode,
            mem_camera_input=cond['mem_camera_input'],
            mem_compress_t=ctx['mem_compress_t'],
            hr_tokens=cond['hr_tokens'], hr_grid_sizes=cond['hr_grid'],
            hr_camera_input=cond['hr_camera_input'],
            hr_t_pos=args.hr_t_pos,
            mem_cfg_drop=args.mem_cfg_drop, hr_cfg_drop=args.hr_cfg_drop,
            cfg_low=args.cfg_low, cfg_high=args.cfg_high,
            adaptive_shift=args.adaptive_shift,
            shift_min=args.shift_min, shift_max=args.shift_max,
        )
        all_videos.append(video.cpu())
        seg_label = instr.replace("+", "-")
        save_segment_incremental(args, ctx, all_videos, pixels, seg_idx, seg_label=seg_label)
        rendered_idx += 1

        current_c2w = abs_c2ws[-1].copy()
        if current_first_frame is not None:
            current_first_frame = video_tensor_to_last_frame_pil(video)
        context_pixel_segments.append(video[:, 1:].cpu() if current_first_frame is not None else video.cpu())
        if current_first_frame is not None:
            context_c2ws.extend(abs_c2ws[1:])
        else:
            context_c2ws.extend(abs_c2ws)


def read_prompt(args):
    if args.prompt is not None:
        return args.prompt
    # Try sibling prompt.txt
    if os.path.isdir(args.input_path):
        ppath = os.path.join(args.input_path, 'prompt.txt')
    else:
        ppath = os.path.join(os.path.dirname(args.input_path), 'prompt.txt')
    if os.path.exists(ppath):
        with open(ppath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    logger.warning("No prompt provided and no prompt.txt found; using default.")
    return 'A high quality video.'


def save_context_and_prompt(args, ctx, context_pixels, prompt):
    """One-time setup: save context.mp4 and prompt.txt to the output dir."""
    config = ctx['config']
    os.makedirs(args.output_dir, exist_ok=True)
    save_video(
        tensor=context_pixels[None],
        save_file=os.path.join(args.output_dir, 'context.mp4'),
        fps=config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
    )
    with open(os.path.join(args.output_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(prompt + '\n')


def save_segment_incremental(args, ctx, videos_so_far, context_pixels, seg_idx, seg_label=None):
    """Save the just-finished segment and rewrite final.mp4 with everything so far.

    Each call writes:
      - segment_<seg_idx>.mp4 (the new segment)
      - final.mp4 (overwritten): context + all segments-so-far (overlap dropped)
    """
    config = ctx['config']
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the new segment
    new_seg = videos_so_far[-1]
    if seg_label:
        seg_fname = f'segment_{seg_idx:03d}_{seg_label}.mp4'
    else:
        seg_fname = f'segment_{seg_idx:03d}.mp4'
    save_video(
        tensor=new_seg[None],
        save_file=os.path.join(args.output_dir, seg_fname),
        fps=config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
    )

    # Rewrite final.mp4 = context + segments-so-far (drop first frame on overlap)
    frames = [context_pixels.cpu()]
    for i, v in enumerate(videos_so_far):
        seg = v.cpu()
        if i > 0 or len(videos_so_far) == 1:
            # Drop overlap frame: subsequent segments share their first frame
            # with the previous segment's last frame; for a single-segment
            # output we also drop one frame to avoid duplicating the context's
            # last frame (which equals the segment's first frame in I2V).
            seg = seg[:, 1:]
        frames.append(seg)
    final = torch.cat(frames, dim=1)
    save_video(
        tensor=final[None],
        save_file=os.path.join(args.output_dir, 'final.mp4'),
        fps=config.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
    )
    logger.info(f"  [incremental] {seg_fname} + final.mp4 "
                f"({final.shape[1]} frames) -> {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CaR: Unified Camera/Action/HardCut/Continue Inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--mode', required=True,
                   choices=['camera', 'action', 'hardcut', 'continue'],
                   help="camera   : I2V with explicit pose sequence\n"
                        "action   : I2V with WASD action commands\n"
                        "hardcut  : I2V with WASD + skip: for hard cuts\n"
                        "continue : V2V continuation from context video")
    p.add_argument('--input_path', required=True,
                   help='Image (.png/.jpg/...) or video (.mp4/...) or folder of images.')
    p.add_argument('--prompt', default=None,
                   help='Text prompt. If omitted, looks for prompt.txt next to input.')
    p.add_argument('--output_dir', default='output/run')

    # Mode-specific inputs
    p.add_argument('--target_poses', default=None,
                   help='Pose file (.json or .npy) for target generation. '
                        'Required for camera mode.')
    p.add_argument('--context_poses', default=None,
                   help='Pose file matching --input_path frames. Required for continue mode.')
    p.add_argument('--motion_sequence', default=None,
                   help='Comma-separated WASD commands. Required for action/hardcut/continue. '
                        'Use "skip:cmd" for hard-cut transitions in hardcut mode. '
                        'Composite commands joined by "+", e.g. "right+down".')
    p.add_argument('--step_size', type=float, default=4.0)
    p.add_argument('--rotate_angle', type=float, default=30.0)
    p.add_argument('--pitch_angle', type=float, default=15.0)

    # Model paths
    p.add_argument('--checkpoint_dir', required=True,
                   help='Wan2.2-TI2V-5B base checkpoint dir.')
    p.add_argument('--car_checkpoint', default=None,
                   help='Trained CaR checkpoint (safetensors / DeepSpeed dir).')

    # Camera
    p.add_argument('--x_fov', type=float, default=100.0)
    p.add_argument('--xi', type=float, default=0.0)
    p.add_argument('--camera_condition', default='relray_absmap')
    p.add_argument('--adaptation_method', default='parallel')
    p.add_argument('--attn_compress', type=int, default=8)
    p.add_argument('--vae_downscale', type=int, default=16)

    # Generation
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--frame_num', type=int, default=81)
    p.add_argument('--sampling_steps', type=int, default=50)
    p.add_argument('--guide_scale', type=float, default=3.0)
    p.add_argument('--shift', type=float, default=5.0)
    p.add_argument('--adaptive_shift', action='store_true', default=True)
    p.add_argument('--no_adaptive_shift', dest='adaptive_shift', action='store_false')
    p.add_argument('--shift_min', type=float, default=3.5)
    p.add_argument('--shift_max', type=float, default=8.0)
    p.add_argument('--sample_solver', default='unipc', choices=['unipc', 'dpm++'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cfg_low', type=float, default=0.0)
    p.add_argument('--cfg_high', type=float, default=1.0)

    # Memory / HR / RoPE
    p.add_argument('--enable_memory', action='store_true', default=True)
    p.add_argument('--no_memory', dest='enable_memory', action='store_false')
    p.add_argument('--memory_in_channels', type=int, default=48)
    p.add_argument('--memory_out_dim', type=int, default=3072)
    p.add_argument('--memory_compression_rate', default='2x4x4')
    p.add_argument('--memory_hidden_channels', default='128,256,512,1024')
    p.add_argument('--memory_num_heads', type=int, default=8)
    p.add_argument('--memory_use_ucpe', action='store_true', default=False)
    p.add_argument('--memory_use_lr_branch', action='store_true', default=True)
    p.add_argument('--rope_mode', default='rope+memrope')

    p.add_argument('--enable_hr_frame', action='store_true', default=True)
    p.add_argument('--no_hr_frame', dest='enable_hr_frame', action='store_false')
    p.add_argument('--hr_frame_mode', default='first', choices=['first', 'closest'])
    p.add_argument('--hr_t_pos', type=float, default=-4200.0)

    p.add_argument('--mem_cfg_drop', action='store_true', default=False)
    p.add_argument('--hr_cfg_drop', action='store_true', default=False)

    return p.parse_args()


def main():
    args = parse_args()
    validate_inputs(args)

    device = torch.device('cuda')
    ctx = load_models(args, device)

    if args.mode == 'camera':
        run_camera_mode(args, ctx)
    elif args.mode in ('action', 'hardcut'):
        run_action_or_hardcut_mode(args, ctx)
    elif args.mode == 'continue':
        run_continue_mode(args, ctx)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
