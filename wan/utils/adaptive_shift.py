"""Adaptive shift computation for flow-matching timestep scheduling.

Shared between training and inference to ensure train-test consistency.
Higher resolution / more frames -> larger shift (emphasize high-noise training).
Lower resolution / fewer frames -> smaller shift (emphasize fine details).
"""

import math


def compute_adaptive_shift(height, width, num_latent_frames,
                           shift_min=3.5, shift_max=8.0,
                           latent_frames_min=5, latent_frames_max=63):
    """Compute adaptive shift based on spatial resolution and temporal length.

    Uses linear interpolation between [latent_frames_min, latent_frames_max]
    to map to [shift_min, shift_max]. Values outside the range are clamped.

    The function is deterministic, ensuring that training and inference use
    the same shift for the same input dimensions.

    Reference points (at 480x832):
        - 17 raw frames  (5 latent)  -> shift = 3.5 (shift_min)
        - 81 raw frames  (21 latent) -> shift ≈ 4.95
        - 249 raw frames (63 latent) -> shift = 8.0 (shift_max)

    Args:
        height: Pixel height of the video (reserved for future multi-res support).
        width: Pixel width of the video (reserved for future multi-res support).
        num_latent_frames: Number of latent temporal frames (after VAE stride).
        shift_min: Minimum shift value (lower bound, default 3.5).
        shift_max: Maximum shift value (upper bound, default 8.0).
        latent_frames_min: Latent frame count that maps to shift_min (default 5).
        latent_frames_max: Latent frame count that maps to shift_max (default 63).

    Returns:
        float: Adaptive shift value clamped to [shift_min, shift_max].
    """
    alpha = (num_latent_frames - latent_frames_min) / (latent_frames_max - latent_frames_min)
    alpha = max(0.0, min(1.0, alpha))
    shift = shift_min + alpha * (shift_max - shift_min)
    return shift
