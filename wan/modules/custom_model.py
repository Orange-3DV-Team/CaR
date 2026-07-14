# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def memory_rope_params(num_frames, dim, theta=10000, compress_t=1):
    """
    Compute RoPE parameters for memory tokens with negative temporal positions.
    
    Temporal positions are scaled by compress_t to match DiT coordinate space:
        [-num_frames * compress_t, ..., -2 * compress_t, -compress_t]
    
    When compress_t=1: [-N, -N+1, ..., -1]  (original behavior)
    When compress_t=2: [-2N, -2N+2, ..., -4, -2]
    
    This ensures memory token temporal spacing matches the actual temporal
    compression ratio from the Memory Encoder, consistent with memrope mode
    in _compute_token_positions.
    
    Args:
        num_frames: Number of memory temporal frames (after compression)
        dim: Dimension for frequency computation (same as rope_params)
        theta: Base frequency (default 10000)
        compress_t: Temporal compression ratio from Memory Encoder (default 1)
    
    Returns:
        freqs: Complex frequency tensor [num_frames, dim//2]
    """
    assert dim % 2 == 0
    # positions = torch.arange(-num_frames, 0) * compress_t  # [-N*ct, ..., -2*ct, -ct]
    positions = -4000 + torch.arange(num_frames) * compress_t  # [-4000, -4000+ct, -4000+2*ct, ...]
    freqs = torch.outer(
        positions.to(torch.float64),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def memory_rope_apply(x, mem_grid_sizes, mem_freqs, spatial_freqs, dit_grid_sizes=None, theta=10000):
    """
    Apply RoPE to memory tokens using negative temporal positions.
    
    Memory tokens use negative temporal positions (from memory_rope_params)
    and spatial positions mapped to DiT coordinate space.
    
    When dit_grid_sizes is provided, memory spatial positions are scaled to
    match DiT coordinate space (consistent with memrope mode in
    _compute_token_positions), placing each memory token at the center of
    its corresponding compressed patch in DiT space.
    
    Args:
        x: [B, mem_seq_len, num_heads, head_dim] memory token q or k
        mem_grid_sizes: [B, 3] containing (mem_f, mem_h, mem_w)
        mem_freqs: Complex frequency tensor for temporal dim [num_frames, d_t//2]
            computed by memory_rope_params with negative positions
        spatial_freqs: Standard rope_params freqs [1024, (d_h + d_w)//2]
            for spatial dimensions (fallback when dit_grid_sizes is None)
        dit_grid_sizes: [B, 3] DiT grid sizes (dit_F, dit_H, dit_W), optional.
            When provided, memory spatial positions are scaled to DiT space.
        theta: Base frequency for RoPE (default 10000, must match rope_params)
    
    Returns:
        x with RoPE applied: [B, mem_seq_len, num_heads, head_dim]
    """
    n, c = x.size(2), x.size(3) // 2
    
    # Split spatial freqs into h and w components
    d_h = c // 3
    d_w = c // 3
    h_freqs, w_freqs = spatial_freqs.split([d_h, d_w], dim=1)
    
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(mem_grid_sizes.tolist()):
        seq_len = f * h * w
        
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # temporal: use negative-position freqs (already scaled by compress_t)
        freqs_t = mem_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1)
        
        # spatial: map memory positions to DiT coordinate space
        if dit_grid_sizes is not None:
            dit_h = int(dit_grid_sizes[i][1].item())
            dit_w = int(dit_grid_sizes[i][2].item())
            h_step = dit_h / h  # e.g., 30/15 = 2.0
            w_step = dit_w / w  # e.g., 52/26 = 2.0
            # Scaled positions: center of each compressed patch in DiT space
            h_positions = torch.arange(h, dtype=torch.float64, device=x.device) * h_step + (h_step - 1) / 2
            w_positions = torch.arange(w, dtype=torch.float64, device=x.device) * w_step + (w_step - 1) / 2
            # Compute freqs at non-integer positions using same theta as rope_params
            h_inv_freq = 1.0 / torch.pow(
                theta, torch.arange(0, d_h * 2, 2, dtype=torch.float64, device=x.device).div(d_h * 2))
            w_inv_freq = 1.0 / torch.pow(
                theta, torch.arange(0, d_w * 2, 2, dtype=torch.float64, device=x.device).div(d_w * 2))
            freqs_h_i = torch.polar(
                torch.ones(h, d_h, dtype=torch.float64, device=x.device),
                torch.outer(h_positions, h_inv_freq))
            freqs_w_i = torch.polar(
                torch.ones(w, d_w, dtype=torch.float64, device=x.device),
                torch.outer(w_positions, w_inv_freq))
            freqs_h_i = freqs_h_i.view(1, h, 1, -1).expand(f, h, w, -1)
            freqs_w_i = freqs_w_i.view(1, 1, w, -1).expand(f, h, w, -1)
        else:
            # Fallback: use standard positive-position freqs (original behavior)
            freqs_h_i = h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1)
            freqs_w_i = w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        
        freqs_i = torch.cat([freqs_t, freqs_h_i, freqs_w_i], dim=-1).reshape(seq_len, 1, -1)
        
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


@torch.amp.autocast('cuda', enabled=False)
def hr_rope_apply(x, hr_grid_sizes, freqs, theta=10000, hr_t_pos=-4200.0):
    """
    Apply RoPE to HR frame tokens at fixed temporal position t=-1.
    
    HR tokens have full DiT spatial resolution and a single temporal frame
    at position t=-1 (between compressed memory tokens at t=-2 and DiT at t=0).
    Spatial positions are identical to DiT tokens: [0, 1, ..., H-1] x [0, 1, ..., W-1].
    
    Args:
        x: [B, hr_seq_len, num_heads, head_dim] HR token q or k
        hr_grid_sizes: [B, 3] containing (1, H, W) — same H, W as DiT after patchify
        freqs: Standard rope_params freqs [1024, C / num_heads / 2] (same as DiT)
        theta: Base frequency (default 10000, must match rope_params)
    
    Returns:
        x with RoPE applied: [B, hr_seq_len, num_heads, head_dim]
    """
    n, c = x.size(2), x.size(3) // 2
    
    # Split freq dimensions (must match rope_apply exactly)
    d_t = c - 2 * (c // 3)
    d_h = c // 3
    d_w = c // 3
    _, h_freqs, w_freqs = freqs.split([d_t, d_h, d_w], dim=1)
    
    output = []
    for i, (f, h, w) in enumerate(hr_grid_sizes.tolist()):
        seq_len = f * h * w
        
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # Temporal: fixed t=hr_t_pos position (must be on same device as freqs)
        device = freqs.device
        t_pos = torch.tensor([hr_t_pos], dtype=torch.float64, device=device)
        t_inv_freq = 1.0 / torch.pow(
            theta, torch.arange(0, d_t * 2, 2, dtype=torch.float64, device=device).div(d_t * 2))
        freqs_t_hr = torch.polar(
            torch.ones(1, d_t, dtype=torch.float64, device=device),
            torch.outer(t_pos, t_inv_freq))
        freqs_t = freqs_t_hr.view(1, 1, 1, -1).expand(f, h, w, -1)
        
        # Spatial: same integer positions as DiT [0, 1, ..., H-1] x [0, 1, ..., W-1]
        freqs_h = h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_w = w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        
        freqs_i = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1).reshape(seq_len, 1, -1)
        
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(x.dtype)

def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        dtype = x.dtype
        x = x.float()
        weight = self.weight.float() if self.elementwise_affine else self.weight
        bias = self.bias.float() if self.elementwise_affine else self.bias

        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps).to(dtype)

        # return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs,
                memory_len=0, mem_grid_sizes=None, mem_freqs=None,
                mem_spatial_freqs=None, rope_mode='rope',
                control_camera_input=None, mem_compress_t=1,
                block_mask=None,
                hr_len=0, hr_grid_sizes=None, hr_camera_input=None,
                hr_t_pos=-4200.0):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] where L may include memory + hr tokens prepended
            seq_lens(Tensor): Shape [B], total sequence lengths (memory + dit, excluding hr)
            grid_sizes(Tensor): Shape [B, 3], dit grid sizes (F, H, W)
            freqs(Tensor): Rope freqs for dit tokens, shape [1024, C / num_heads / 2]
            memory_len(int): Number of memory tokens prepended to the sequence (0 if none)
            mem_grid_sizes(Tensor): Shape [B, 3], memory grid sizes (mem_F, mem_H, mem_W)
            mem_freqs(Tensor): Memory temporal RoPE freqs (negative positions)
            mem_spatial_freqs(Tensor): Memory spatial RoPE freqs (standard positive positions)
            rope_mode(str): Position encoding mode ('rope', 'prope', 'memrope')
                - rope: standard RoPE in self-attn (memory negative t, dit positive t)
                - prope: skip standard RoPE, position encoding handled by cam_self_attn
                - memrope: skip standard RoPE, position encoding handled by cam_self_attn
            block_mask(BlockMask, optional): FlexAttention block mask for causal mode.
                When provided, uses FlexAttention instead of Flash Attention.
            hr_len(int): Number of HR frame tokens inserted between memory and dit (0 if none)
            hr_grid_sizes(Tensor): Shape [B, 3], HR frame grid sizes (1, H_patch, W_patch)
                Same spatial resolution as DiT after patchify.
            hr_camera_input(dict, optional): Camera control parameters for HR frame tokens
                (for memrope mode). Contains 'viewmats' [B, N_hr, 4, 4].
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if rope_mode == 'rope':
            # Standard RoPE mode: apply RoPE in self-attention
            if memory_len > 0 or hr_len > 0:
                # Split: [memory (N_mem) | hr (N_hr) | dit (rest)]
                q_mem = q[:, :memory_len] if memory_len > 0 else None
                k_mem = k[:, :memory_len] if memory_len > 0 else None
                v_mem = v[:, :memory_len] if memory_len > 0 else None
                
                q_hr = q[:, memory_len:memory_len + hr_len] if hr_len > 0 else None
                k_hr = k[:, memory_len:memory_len + hr_len] if hr_len > 0 else None
                v_hr = v[:, memory_len:memory_len + hr_len] if hr_len > 0 else None
                
                q_dit = q[:, memory_len + hr_len:]
                k_dit = k[:, memory_len + hr_len:]
                v_dit = v[:, memory_len + hr_len:]
                
                # Apply negative-position RoPE to memory tokens
                if memory_len > 0 and mem_grid_sizes is not None and mem_freqs is not None:
                    q_mem = memory_rope_apply(q_mem, mem_grid_sizes, mem_freqs, mem_spatial_freqs, dit_grid_sizes=grid_sizes)
                    k_mem = memory_rope_apply(k_mem, mem_grid_sizes, mem_freqs, mem_spatial_freqs, dit_grid_sizes=grid_sizes)
                
                # Apply t=-1 RoPE to HR frame tokens (full DiT spatial resolution)
                if hr_len > 0 and hr_grid_sizes is not None:
                    q_hr = hr_rope_apply(q_hr, hr_grid_sizes, freqs, hr_t_pos=hr_t_pos)
                    k_hr = hr_rope_apply(k_hr, hr_grid_sizes, freqs, hr_t_pos=hr_t_pos)
                
                # Apply standard RoPE to dit tokens (positions start from 0)
                q_dit = rope_apply(q_dit, grid_sizes, freqs)
                k_dit = rope_apply(k_dit, grid_sizes, freqs)
                
                # Concatenate back: [memory | hr | dit]
                q_parts = [p for p in [q_mem, q_hr, q_dit] if p is not None]
                k_parts = [p for p in [k_mem, k_hr, k_dit] if p is not None]
                v_parts = [p for p in [v_mem, v_hr, v_dit] if p is not None]
                q = torch.cat(q_parts, dim=1)
                k = torch.cat(k_parts, dim=1)
                v = torch.cat(v_parts, dim=1)
            else:
                # No memory or HR tokens, standard RoPE
                q = rope_apply(q, grid_sizes, freqs)
                k = rope_apply(k, grid_sizes, freqs)
        elif rope_mode in ('prope', 'memrope'):
            # PRoPE mode: apply camera projection + spatial/temporal RoPE in self-attention
            if hasattr(self, 'prope_attn') and self.prope_attn is not None and control_camera_input is not None:
                from wan.modules.prope import _rope_precompute_coeffs
                from wan.modules.ucpe_camera_control import _compute_token_positions

                # Reshape q, k, v to [B, H, T, D_head] for PRoPE
                q = q.transpose(1, 2)  # [B, H, S, D]
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                # Compute RoPE coefficients for memrope mode
                coeffs_t = None
                coeffs_x_override = None
                coeffs_y_override = None
                if rope_mode == 'memrope':
                    d_x = (d // 6) // 2 * 2  # round down to even
                    d_y = (d // 6) // 2 * 2
                    d_t = d - d // 2 - d_x - d_y

                    all_x, all_y, all_t = _compute_token_positions(
                        T=s, memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                        patches_x=self.prope_attn.patches_x,
                        patches_y=self.prope_attn.patches_y,
                        mem_compress_t=mem_compress_t,
                        hr_len=hr_len, hr_grid_sizes=hr_grid_sizes,
                        device=x.device,
                    )
                    coeffs_x_override = _rope_precompute_coeffs(
                        all_x, freq_base=100.0, freq_scale=1.0, feat_dim=d_x, dtype=x.dtype)
                    coeffs_y_override = _rope_precompute_coeffs(
                        all_y, freq_base=100.0, freq_scale=1.0, feat_dim=d_y, dtype=x.dtype)
                    coeffs_t = _rope_precompute_coeffs(
                        all_t, freq_base=100.0, freq_scale=1.0, feat_dim=d_t, dtype=x.dtype)

                # Build viewmats for self_attn (need combined memory + dit viewmats)
                viewmats = control_camera_input["viewmats"].to(dtype=x.dtype)
                Ks = control_camera_input.get("K", None)
                if Ks is not None:
                    Ks = Ks.to(dtype=x.dtype)

                self.prope_attn._precompute_and_cache_apply_fns(
                    viewmats=viewmats, Ks=Ks,
                    coeffs_x=coeffs_x_override if coeffs_x_override is not None else control_camera_input.get("coeffs_x", None),
                    coeffs_y=coeffs_y_override if coeffs_y_override is not None else control_camera_input.get("coeffs_y", None),
                    rope_mode=rope_mode,
                    coeffs_t=coeffs_t,
                )

                q = self.prope_attn._apply_to_q(q)
                k = self.prope_attn._apply_to_kv(k)
                v = self.prope_attn._apply_to_kv(v)

                # Reshape back to [B, S, H, D] for flash_attention
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
            # else: no prope_attn available, skip RoPE (fallback to old behavior)
        # else: unknown rope_mode, skip RoPE

        if block_mask is not None:
            # Causal mode: use FlexAttention with BlockMask
            from wan.modules.causal_mask import flex_attention_compiled
            import math as _math

            # FlexAttention expects [B, num_heads, S, head_dim]
            q = q.transpose(1, 2)  # [B, H, S, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Pad sequence length to multiple of 128 (FlexAttention requirement)
            actual_seq_len = q.shape[2]
            padded_length = _math.ceil(actual_seq_len / 128) * 128 - actual_seq_len

            if padded_length > 0:
                pad_shape = (q.shape[0], q.shape[1], padded_length, q.shape[3])
                q = torch.cat([q, torch.zeros(pad_shape, dtype=q.dtype, device=q.device)], dim=2)
                k = torch.cat([k, torch.zeros(pad_shape, dtype=k.dtype, device=k.device)], dim=2)
                v = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype, device=v.device)], dim=2)

            x = flex_attention_compiled(q, k, v, block_mask=block_mask)

            # Remove padding
            if padded_length > 0:
                x = x[:, :, :actual_seq_len, :]

            # Transpose back to [B, S, H, D]
            x = x.transpose(1, 2)
        else:
            # Default mode: use Flash Attention
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # Apply inverse transform for PRoPE if applicable
        if rope_mode in ('prope', 'memrope') and hasattr(self, 'prope_attn') and self.prope_attn is not None and control_camera_input is not None:
            # x shape after flash_attention: [B, S, H, D]
            x = x.transpose(1, 2)  # [B, H, S, D]
            x = self.prope_attn._apply_to_o(x)
            x = x.transpose(1, 2)  # [B, S, H, D]

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        control_camera_input=None,
        memory_len=0,
        mem_grid_sizes=None,
        mem_freqs=None,
        mem_spatial_freqs=None,
        rope_mode='rope',
        mem_camera_input=None,
        mem_compress_t=1,
        block_mask=None,
        hr_len=0,
        hr_grid_sizes=None,
        hr_camera_input=None,
        hr_t_pos=-4200.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] where L may include memory + hr tokens prepended
            e(Tensor): Shape [B, L, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence (dit only, without memory/hr)
            grid_sizes(Tensor): Shape [B, 3], dit grid sizes (F, H, W)
            freqs(Tensor): Rope freqs for dit tokens, shape [1024, C / num_heads / 2]
            control_camera_input: Camera control parameters for dit tokens (optional)
            memory_len(int): Number of memory tokens prepended (0 if none)
            mem_grid_sizes(Tensor): Shape [B, 3], memory grid sizes
            mem_freqs(Tensor): Memory temporal RoPE freqs (negative positions)
            mem_spatial_freqs(Tensor): Memory spatial RoPE freqs
            rope_mode(str): Position encoding mode ('dit_mode+ucpe_mode' or single mode)
            mem_camera_input(dict): Camera control parameters for memory tokens (optional)
                Contains 'viewmats' [B, T_mem*H*W, 4, 4] and optionally 'K' [B, T_mem, 3, 3]
            mem_compress_t(int): Temporal compression ratio from memory encoder (default 1)
            block_mask(BlockMask, optional): FlexAttention block mask for causal mode.
                When provided, self_attn uses FlexAttention instead of Flash Attention.
                Only affects self-attention, not cross-attention.
            hr_len(int): Number of HR frame tokens inserted between memory and dit (0 if none)
            hr_grid_sizes(Tensor): Shape [B, 3], HR frame grid sizes (1, H_patch, W_patch)
            hr_camera_input(dict, optional): Camera control parameters for HR frame tokens
                (for memrope mode). Contains 'viewmats' [B, N_hr, 4, 4].
        """
        # Parse composite rope_mode: "dit_mode+ucpe_mode" or single mode
        if '+' in rope_mode:
            dit_rope_mode, ucpe_rope_mode = rope_mode.split('+', 1)
        else:
            dit_rope_mode = rope_mode
            ucpe_rope_mode = rope_mode

        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        input_x = self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)

        # Compute augmented seq_lens that includes memory + hr tokens
        aug_seq_lens = seq_lens
        if memory_len > 0:
            aug_seq_lens = aug_seq_lens + memory_len
        if hr_len > 0:
            aug_seq_lens = aug_seq_lens + hr_len

        # Helper: build combined camera input for all tokens (memory + hr + dit)
        def _build_combined_camera_input(dit_cam_input, mem_cam_input, memory_len,
                                         hr_cam_input=None, hr_len=0):
            """Concatenate memory, hr, and dit viewmats for cam_self_attn on all tokens."""
            if (mem_cam_input is None or memory_len == 0) and (hr_cam_input is None or hr_len == 0):
                return dit_cam_input
            combined = {}
            for key in dit_cam_input:
                parts = []
                if key == 'viewmats':
                    if mem_cam_input is not None and memory_len > 0 and key in mem_cam_input:
                        parts.append(mem_cam_input[key])
                    if hr_cam_input is not None and hr_len > 0 and key in hr_cam_input:
                        parts.append(hr_cam_input[key])
                    parts.append(dit_cam_input[key])
                    combined[key] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                elif key == 'K':
                    if mem_cam_input is not None and memory_len > 0 and key in mem_cam_input:
                        parts.append(mem_cam_input[key])
                    if hr_cam_input is not None and hr_len > 0 and key in hr_cam_input:
                        parts.append(hr_cam_input[key])
                    parts.append(dit_cam_input[key])
                    combined[key] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                elif key == 'cam_emb':
                    if mem_cam_input is not None and memory_len > 0 and key in mem_cam_input:
                        parts.append(mem_cam_input[key])
                    if hr_cam_input is not None and hr_len > 0 and key in hr_cam_input:
                        parts.append(hr_cam_input[key])
                    parts.append(dit_cam_input[key])
                    combined[key] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                elif key == 'coeffs_x' or key == 'coeffs_y':
                    # These are spatial coefficients, shared between memory and dit
                    combined[key] = dit_cam_input[key]
                else:
                    combined[key] = dit_cam_input[key]
            return combined

        # self-attention + camera control
        # dit_rope_mode controls self_attn, ucpe_rope_mode controls cam_self_attn
        if control_camera_input is not None:
            if hasattr(self, "cam_encoder") and hasattr(self, "projector"):
                # cam_emb additive embedding: only used in rope mode (relray_absmap etc.)
                # In prope/memrope modes, camera info is encoded via PRoPE rotary encoding.
                if dit_rope_mode == 'rope' and "cam_emb" in control_camera_input:
                    cam_emb = control_camera_input["cam_emb"]
                    y = self.cam_encoder(cam_emb)
                    # input_x layout: [memory | HR | dit]
                    if memory_len > 0 or hr_len > 0:
                        T_total = input_x.shape[1]
                        dit_tokens = T_total - memory_len - hr_len
                        dit_f_grid = grid_sizes[0, 0].item()
                        dit_h = grid_sizes[0, 1].item()
                        dit_w = grid_sizes[0, 2].item()
                        hw = dit_h * dit_w

                        if y.shape[1] == dit_tokens:
                            dit_y = y
                        elif y.shape[1] == dit_f_grid:
                            # per-frame: repeat to dit_f * H * W
                            dit_y = repeat(y, "b f d -> b (f hw) d", hw=hw)
                        else:
                            # fallback: infer hw by divisibility
                            inferred_hw = dit_tokens // y.shape[1] if y.shape[1] > 0 else 0
                            if inferred_hw > 0 and inferred_hw * y.shape[1] == dit_tokens:
                                dit_y = repeat(y, "b f d -> b (f hw) d", hw=inferred_hw)
                            else:
                                # keep y so the downstream shape error stays visible
                                dit_y = y

                        # zero-pad memory and HR prefixes
                        prefix_parts = []
                        if memory_len > 0:
                            prefix_parts.append(torch.zeros(
                                dit_y.shape[0], memory_len, dit_y.shape[2],
                                device=dit_y.device, dtype=dit_y.dtype))
                        if hr_len > 0:
                            prefix_parts.append(torch.zeros(
                                dit_y.shape[0], hr_len, dit_y.shape[2],
                                device=dit_y.device, dtype=dit_y.dtype))
                        y = torch.cat(prefix_parts + [dit_y], dim=1)
                    else:
                        # no memory or HR prefix
                        if y.shape[1] != input_x.shape[1]:
                            hw = input_x.shape[1] // cam_emb.shape[1]
                            y = repeat(y, "b f d -> b (f hw) d", hw=hw)
                    input_x = input_x + y
                residual = self.projector(self.self_attn(
                    input_x, aug_seq_lens, grid_sizes, freqs,
                    memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                    mem_freqs=mem_freqs, mem_spatial_freqs=mem_spatial_freqs,
                    rope_mode=dit_rope_mode,
                    control_camera_input=control_camera_input,
                    mem_compress_t=mem_compress_t,
                    block_mask=block_mask,
                    hr_len=hr_len, hr_grid_sizes=hr_grid_sizes,
                    hr_t_pos=hr_t_pos))
            elif hasattr(self, "cam_self_attn"):
                # cam_self_attn operates on ALL tokens (memory + hr + dit)
                # Build combined camera input with memory + hr viewmats prepended
                combined_cam_input = _build_combined_camera_input(
                    control_camera_input, mem_camera_input, memory_len,
                    hr_cam_input=hr_camera_input, hr_len=hr_len)

                # Extract save_attn_step from combined_cam_input if present
                save_attn_step = None
                if isinstance(combined_cam_input, dict):
                    save_attn_step = combined_cam_input.pop('save_attn_step', None)
                
                # Get block_idx from cam_self_attn module
                block_idx = getattr(self.cam_self_attn, 'block_idx', None)

                if self.cam_self_attn.adaptation_method == "before":
                    cam_out = self.cam_self_attn(
                        input_x, combined_cam_input, rope_mode=ucpe_rope_mode,
                        memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                        mem_compress_t=mem_compress_t,
                        save_attn_step=save_attn_step, hr_len=hr_len)
                    input_x = input_x + cam_out
                residual = self.self_attn(
                    input_x, aug_seq_lens, grid_sizes, freqs,
                    memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                    mem_freqs=mem_freqs, mem_spatial_freqs=mem_spatial_freqs,
                    rope_mode=dit_rope_mode,
                    control_camera_input=combined_cam_input,
                    mem_compress_t=mem_compress_t,
                    block_mask=block_mask,
                    hr_len=hr_len, hr_grid_sizes=hr_grid_sizes,
                    hr_t_pos=hr_t_pos)
                if self.cam_self_attn.adaptation_method == "parallel":
                    cam_out = self.cam_self_attn(
                        input_x, combined_cam_input, rope_mode=ucpe_rope_mode,
                        memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                        mem_compress_t=mem_compress_t,
                        save_attn_step=save_attn_step, hr_len=hr_len)
                    residual = residual + cam_out
            else:
                raise NotImplementedError
        else:
            residual = self.self_attn(
                input_x, aug_seq_lens, grid_sizes, freqs,
                memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                mem_freqs=mem_freqs, mem_spatial_freqs=mem_spatial_freqs,
                rope_mode=dit_rope_mode,
                control_camera_input=control_camera_input,
                mem_compress_t=mem_compress_t,
                block_mask=block_mask,
                hr_len=hr_len, hr_grid_sizes=hr_grid_sizes,
                hr_t_pos=hr_t_pos)

        x = x + residual * e[2].squeeze(2)
        if control_camera_input is not None \
            and hasattr(self, "cam_self_attn") \
                and self.cam_self_attn.adaptation_method == "after":
            # cam_self_attn operates on ALL tokens (memory + hr + dit)
            combined_cam_input = _build_combined_camera_input(
                control_camera_input, mem_camera_input, memory_len,
                hr_cam_input=hr_camera_input, hr_len=hr_len)
            
            # Extract save_attn_step from combined_cam_input if present
            save_attn_step = None
            if isinstance(combined_cam_input, dict):
                save_attn_step = combined_cam_input.pop('save_attn_step', None)
            
            cam_out = self.cam_self_attn(
                x, combined_cam_input, rope_mode=ucpe_rope_mode,
                memory_len=memory_len, mem_grid_sizes=mem_grid_sizes,
                mem_compress_t=mem_compress_t,
                save_attn_step=save_attn_step, hr_len=hr_len)
            x = x + cam_out

        # cross-attention & ffn function
        # Memory tokens also participate in cross-attention and FFN
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = (
            self.head(
                self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))

        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 in_dim_control_adapter=24,
                 downscale_factor_control_adapter=16,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.in_dim_control_adapter = in_dim_control_adapter
        self.downscale_factor_control_adapter = downscale_factor_control_adapter

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        
        self.control_adapter = None

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        control_camera_input=None, 
        gradient_checkpointing=False,
        memory_tokens=None,
        mem_grid_sizes=None,
        rope_mode='rope',
        mem_camera_input=None,
        mem_compress_t=1,
        enable_causal=False,
        causal_block_frames=3,
        hr_tokens=None,
        hr_grid_sizes=None,
        hr_camera_input=None,
        hr_t_pos=-4200.0,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding (dit tokens only)
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            gradient_checkpointing (`bool`, *optional*, defaults to False):
            memory_tokens (Tensor, *optional*):
                Compressed memory tokens [B, N_mem, dim], prepended to dit sequence.
                These are clean (no noise added) and participate in all blocks.
            mem_grid_sizes (Tensor, *optional*):
                Memory grid sizes [B, 3] containing (mem_F, mem_H, mem_W)
            rope_mode (str, *optional*, defaults to 'rope'):
                Position encoding mode ('rope', 'prope', 'memrope')
            mem_camera_input (dict, *optional*):
                Camera control parameters for memory tokens, containing 'viewmats' etc.
            enable_causal (bool, *optional*, defaults to False):
                Enable causal block mask for self-attention using FlexAttention.
            causal_block_frames (int, *optional*, defaults to 3):
                Number of temporal frames per causal block.
            hr_tokens (Tensor, *optional*):
                High-resolution frame tokens [B, N_hr, dim] at t=-1 position.
                Encoded via VAE + patch_embedding from a single context frame.
                Inserted between memory and dit tokens.
            hr_grid_sizes (Tensor, *optional*):
                HR frame grid sizes [B, 3] containing (1, H_patch, W_patch).
                Same spatial resolution as DiT (after patchify).
            hr_camera_input (dict, *optional*):
                Camera control parameters for HR frame tokens (for memrope mode).
                Contains 'viewmats' [B, N_hr, 4, 4].

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter
        if self.control_adapter is not None and control_camera_input is not None:
            control_camera = self.control_adapter(control_camera_input)
            x = [u + v for u, v in zip(x, control_camera)]
            control_camera_input = None

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # Memory tokens
        N_mem = 0
        mem_freqs = None
        mem_spatial_freqs = None
        
        if memory_tokens is not None and mem_grid_sizes is not None:
            N_mem = memory_tokens.shape[1]
            total_seq_len = seq_len + N_mem
            
            # Compute memory RoPE freqs (negative temporal positions)
            mem_f = mem_grid_sizes[0, 0].item()
            d = self.dim // self.num_heads  # head_dim = 128
            d_t = d - 4 * (d // 6)  # temporal freq dim = 44
            d_h = 2 * (d // 6)       # spatial h freq dim = 42
            d_w = 2 * (d // 6)       # spatial w freq dim = 42
            
            # Temporal: negative positions scaled by compress_t [-mem_f*ct, ..., -ct]
            mem_freqs = memory_rope_params(mem_f, d_t, compress_t=mem_compress_t).to(device)
            # Spatial: standard positive positions (shared with dit)
            mem_spatial_freqs = torch.cat([
                rope_params(1024, d_h),
                rope_params(1024, d_w),
            ], dim=1).to(device)
            
            # Prepend memory tokens to dit sequence
            # memory_tokens: [B, N_mem, dim], x: [B, seq_len, dim]
            memory_tokens = memory_tokens.to(dtype=x.dtype)
            x = torch.cat([memory_tokens, x], dim=1)  # [B, N_mem + seq_len, dim]
        else:
            total_seq_len = seq_len

        # HR frame tokens
        N_hr = 0
        if hr_tokens is not None and hr_grid_sizes is not None:
            N_hr = hr_tokens.shape[1]
            hr_tokens = hr_tokens.to(dtype=x.dtype)
            # Insert HR tokens between memory and dit tokens
            # Current order: [memory_tokens (N_mem) | dit_tokens (seq_len)]
            # New order:     [memory_tokens (N_mem) | hr_tokens (N_hr) | dit_tokens (seq_len)]
            if N_mem > 0:
                x = torch.cat([x[:, :N_mem], hr_tokens, x[:, N_mem:]], dim=1)
            else:
                x = torch.cat([hr_tokens, x], dim=1)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)

        bt = t.size(0)
        t_flat = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t_flat).unflatten(0, (bt, seq_len)).to(self.dtype))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # Memory tokens use timestep=0 embedding (clean reference)
        if N_mem > 0:
            mem_t = torch.zeros(bt, N_mem, device=device, dtype=t.dtype)
            mem_e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        mem_t.flatten()).unflatten(0, (bt, N_mem)).to(self.dtype))
            mem_e0 = self.time_projection(mem_e).unflatten(2, (6, self.dim))
            
            # Prepend memory time embeddings
            e = torch.cat([mem_e, e], dim=1)    # [B, N_mem + seq_len, dim]
            e0 = torch.cat([mem_e0, e0], dim=1)  # [B, N_mem + seq_len, 6, dim]

        # HR frame tokens use timestep=0 embedding (clean reference, same as memory)
        if N_hr > 0:
            hr_t = torch.zeros(bt, N_hr, device=device, dtype=t.dtype)
            hr_e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        hr_t.flatten()).unflatten(0, (bt, N_hr)).to(self.dtype))
            hr_e0 = self.time_projection(hr_e).unflatten(2, (6, self.dim))
            
            # Insert HR time embeddings between memory and dit
            # Current e order: [mem_e (N_mem) | dit_e (seq_len)]
            # New e order:     [mem_e (N_mem) | hr_e (N_hr) | dit_e (seq_len)]
            if N_mem > 0:
                e = torch.cat([e[:, :N_mem], hr_e, e[:, N_mem:]], dim=1)
                e0 = torch.cat([e0[:, :N_mem], hr_e0, e0[:, N_mem:]], dim=1)
            else:
                e = torch.cat([hr_e, e], dim=1)
                e0 = torch.cat([hr_e0, e0], dim=1)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Causal block mask
        block_mask = None
        if enable_causal:
            from wan.modules.causal_mask import create_causal_block_mask, print_causal_block_mask

            # Get dit grid dimensions
            num_frames = grid_sizes[0, 0].item()  # F
            frame_seqlen = grid_sizes[0, 1].item() * grid_sizes[0, 2].item()  # H * W

            # Cache block mask to avoid re-creation every forward pass
            cache_key = (num_frames, frame_seqlen, causal_block_frames, N_mem)
            if not hasattr(self, '_cached_block_mask') or \
               not hasattr(self, '_cached_block_mask_key') or \
               self._cached_block_mask_key != cache_key:
                block_mask = create_causal_block_mask(
                    device=device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=causal_block_frames,
                    memory_len=N_mem,
                )
                self._cached_block_mask = block_mask
                self._cached_block_mask_key = cache_key

                # Print mask info on first creation (for debugging)
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print_causal_block_mask(
                        block_mask,
                        num_frames=num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=causal_block_frames,
                        memory_len=N_mem,
                    )
            else:
                block_mask = self._cached_block_mask

        # arguments for blocks
        args = (
            e0,
            seq_lens,
            grid_sizes,
            self.freqs,
            context,
            context_lens, 
            control_camera_input,
            N_mem,
            mem_grid_sizes,
            mem_freqs,
            mem_spatial_freqs,
            rope_mode,
            mem_camera_input,
            mem_compress_t,
            block_mask,
            N_hr,
            hr_grid_sizes,
            hr_camera_input,
            hr_t_pos,
        )

        for block in self.blocks:
            if gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, *args, use_reentrant=False)
            else:
                x = block(x, *args)

        # Remove extra tokens before head
        # Remove HR tokens first (between memory and dit)
        if N_hr > 0:
            x = torch.cat([x[:, :N_mem], x[:, N_mem + N_hr:]], dim=1)
            e = torch.cat([e[:, :N_mem], e[:, N_mem + N_hr:]], dim=1)
        # Then remove memory tokens
        if N_mem > 0:
            x = x[:, N_mem:]      # Remove memory tokens
            e = e[:, N_mem:]      # Also trim time embedding for head

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
