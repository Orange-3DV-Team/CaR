import torch
from einops import repeat
from functools import lru_cache
from typing import Optional


def create_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create coordinate grid with height and width

    `z-axis` scale is `1`

    params:
    - height (int)
    - width (int)
    - batch (Optional[int])
    - dtype (torch.dtype)
    - device (torch.device)

    return:
    - grid (torch.Tensor)
    """

    # NOTE: RuntimeError: "linspace_cpu" not implemented for Half
    if device.type == "cpu":
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: {dtype} is not supported by {device.type}\n"
            "If device is `cpu`, use float32 or float64"
        )

    _xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    _ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs], indexing="ij")
    zs = torch.ones_like(xs, dtype=dtype, device=device)
    grid = torch.stack((xs, ys, zs), dim=2)
    # grid shape (h, w, 3)

    # batched (stacked copies)
    if batch is not None:
        assert isinstance(
            batch, int
        ), f"ERR: batch needs to be integer: batch={batch}"
        assert (
            batch > 0
        ), f"ERR: batch size needs to be larger than 0: batch={batch}"
        # FIXME: faster way of copying?
        # grid = torch.cat([grid.unsqueeze(0)] * batch)
        grid = repeat(grid, "... -> b ...", b=batch)
        
        # grid shape is (b, h, w, 3)

    return grid


@lru_cache(maxsize=128)
def ucm_unproject_grid(
    height: int,
    width: int,
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    xi: float | torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    y_down: bool = True,
) -> torch.Tensor:
    """Create UCM camera rays from pixel coordinates."""
    fx_, fy_, cx_, cy_, xi_ = fx, fy, cx, cy, xi

    def to_tensor1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx, fy, cx, cy, xi = map(to_tensor1d, (fx, fy, cx, cy, xi))
    B = fx.shape[0]

    grid = create_grid(height=height, width=width, batch=B, dtype=dtype, device=device)
    u = grid[..., 0]
    v = grid[..., 1]

    fx = fx[:, None, None]
    fy = fy[:, None, None]
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    xi = xi[:, None, None]

    x = (u - cx) / fx
    y = (v - cy) / fy
    if not y_down:
        y = -y

    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)

    X = gamma * x
    Y = gamma * y
    Z = gamma - xi

    d_cam = torch.stack([X, Y, Z], dim=-1)  # [B, H, W, 3]

    is_scalar_input = all(not torch.is_tensor(p) for p in (fx_, fy_, cx_, cy_, xi_))
    if is_scalar_input:
        return d_cam[0]  # → [H, W, 3]
    else:
        return d_cam       # → [B, H, W, 3]