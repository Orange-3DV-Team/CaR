"""
Generate a video with keyboard (WASD) and mouse-sphere (rotation) indicator
overlays for the action-mode output of inference.py.

Pairs each segment_NNN.mp4 under --segments_dir with the corresponding
instruction from --motion_sequence (same syntax as run_action.sh, including
composite commands joined with '+', e.g. "w+right"). 'skip:' instructions
are dropped since they do not produce a rendered segment.

Left:  WASD keys with the active key highlighted (no panel background).
Right: glass sphere with crosshair + mouse icon swiping toward the active
       rotation direction (left/right/up/down).

Composite commands (e.g. 'w+right') light up BOTH the WASD key and the
sphere arrow at the same time.
"""

import argparse
import glob
import os

import cv2
import numpy as np


WASD_ACTIONS = {"w", "a", "s", "d"}
ARROW_ACTIONS = {"left", "right", "up", "down"}

# ============== Configuration ==============
FPS = 24

# --- Keyboard Keys Design ---
KEY_SIZE = 54
KEY_GAP = 8
KEY_RADIUS = 12
KEY_BG = (35, 38, 45)           # Dark slate (slightly darker)
KEY_BORDER = (75, 80, 90)       # Subtle lighter border
KEY_ARROW_COLOR = (135, 138, 145)  # Dim gray letters (inactive)
KEY_ACTIVE_BG = (237, 58, 124)   # Anonymous Institution purple (violet-600 BGR)
KEY_ACTIVE_BORDER = (250, 139, 167)
KEY_ACTIVE_ARROW = (255, 255, 255)
KEY_ALPHA = 245                  # Almost opaque so background can't fake active
KEY_ACTIVE_ALPHA = 250

# --- Mouse Sphere Design ---
SPHERE_RADIUS = 78
SPHERE_BG = (120, 125, 130)     # Glass-like gray
SPHERE_ALPHA = 80               # Very translucent
SPHERE_BORDER_COLOR = (160, 165, 170)
SPHERE_BORDER_ALPHA = 140
CROSSHAIR_COLOR = (140, 145, 150)
CROSSHAIR_ALPHA = 100
ARROW_TIP_COLOR = (150, 155, 160)
ARROW_TIP_ACTIVE = (246, 92, 139)   # Bright purple for active direction
MOUSE_SIZE = 27                 # Mouse icon radius
MOUSE_COLOR = (190, 160, 180)   # Muted purple mouse body
MOUSE_ACTIVE_COLOR = (250, 139, 167)  # Bright purple when active
MOUSE_SHIFT = 40                # How far mouse moves from center
MOUSE_SMOOTH_ALPHA = 0.18       # Per-frame easing factor (lower = smoother/slower)

MARGIN_X = 24
MARGIN_BOTTOM = 22

FONT = cv2.FONT_HERSHEY_SIMPLEX


def rounded_rect_mask(w, h, radius):
    """Create a rounded rectangle alpha mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    r = min(radius, w // 2, h // 2)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    return mask


def draw_arrow_icon(img, cx, cy, direction, color, size=13):
    """Draw a filled arrow triangle."""
    s = size
    if direction == "up" or direction == "w":
        pts = np.array([[cx, cy - s], [cx - s, cy + s//2], [cx + s, cy + s//2]])
    elif direction == "down" or direction == "s":
        pts = np.array([[cx, cy + s], [cx - s, cy - s//2], [cx + s, cy - s//2]])
    elif direction == "left" or direction == "a":
        pts = np.array([[cx - s, cy], [cx + s//2, cy - s], [cx + s//2, cy + s]])
    elif direction == "right" or direction == "d":
        pts = np.array([[cx + s, cy], [cx - s//2, cy - s], [cx - s//2, cy + s]])
    else:
        return
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)


DIR_VECTORS = {
    "up":    (0.0, -1.0),
    "down":  (0.0,  1.0),
    "left":  (-1.0, 0.0),
    "right": (1.0,  0.0),
}


def directions_to_vector(directions):
    """Sum unit vectors for every active direction, then normalize to length 1.

    Enables diagonal composite directions such as 'left+up' -> (-0.707, -0.707).
    Returns (0, 0) if no directions are active.
    """
    if not directions:
        return (0.0, 0.0)
    dx = dy = 0.0
    for d in directions:
        vx, vy = DIR_VECTORS.get(d, (0.0, 0.0))
        dx += vx
        dy += vy
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return (0.0, 0.0)
    return (dx / n, dy / n)


def target_mouse_offset(directions):
    """Target (dx, dy) pixel offset for the given active arrow set."""
    vx, vy = directions_to_vector(directions)
    return (vx * MOUSE_SHIFT, vy * MOUSE_SHIFT)


def draw_mouse_icon(img, alpha_ch, cx, cy, color, alpha_val=220):
    """Draw a simple mouse icon (oval with scroll wheel line)."""
    # Mouse body - rounded capsule shape
    mw, mh = 12, 16
    # Draw oval body
    cv2.ellipse(img, (cx, cy), (mw, mh), 0, 0, 360, color, -1, cv2.LINE_AA)
    # Border
    cv2.ellipse(img, (cx, cy), (mw, mh), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    # Middle line (scroll wheel)
    cv2.line(img, (cx, cy - mh + 5), (cx, cy - 2), (220, 220, 220), 1, cv2.LINE_AA)
    # Divider line
    cv2.line(img, (cx - mw, cy - 4), (cx + mw, cy - 4), (200, 200, 200), 1, cv2.LINE_AA)
    # Set alpha for the mouse region
    cv2.ellipse(alpha_ch, (cx, cy), (mw, mh), 0, 0, 360, alpha_val, -1, cv2.LINE_AA)


def build_keyboard_overlay(active_keys):
    """Build WASD keyboard keys overlay (no panel, just floating keys).
    Keys display W/A/S/D letters. active_keys is a set/iterable of keys."""
    if active_keys is None:
        active_keys = set()
    elif isinstance(active_keys, str):
        active_keys = {active_keys}
    else:
        active_keys = set(active_keys)
    step = KEY_SIZE + KEY_GAP
    # Bounding box for cross layout
    ow = 3 * KEY_SIZE + 2 * KEY_GAP + 8  # +padding
    oh = 2 * KEY_SIZE + KEY_GAP + 8
    pad = 4  # internal offset

    bgr = np.zeros((oh, ow, 3), dtype=np.uint8)
    alpha = np.zeros((oh, ow), dtype=np.uint8)

    # Key positions (cross): W on top-center, A/S/D on bottom row
    positions = {
        "w": (pad + step, pad),
        "a": (pad, pad + step),
        "s": (pad + step, pad + step),
        "d": (pad + 2 * step, pad + step),
    }

    for key_name, (kx, ky) in positions.items():
        is_active = (key_name in active_keys)
        ks = KEY_SIZE
        kr = KEY_RADIUS

        # Key mask
        key_mask = rounded_rect_mask(ks, ks, kr)

        if is_active:
            bg_color = KEY_ACTIVE_BG
            border_color = KEY_ACTIVE_BORDER
            text_color = KEY_ACTIVE_ARROW
            a_val = KEY_ACTIVE_ALPHA
        else:
            bg_color = KEY_BG
            border_color = KEY_BORDER
            text_color = KEY_ARROW_COLOR
            a_val = KEY_ALPHA

        # Draw key background
        key_patch = np.zeros((ks, ks, 3), dtype=np.uint8)
        key_patch[:] = bg_color

        # Draw border (2px)
        border_m = rounded_rect_mask(ks, ks, kr)
        inner_m = np.zeros_like(border_m)
        if ks > 4:
            inner_m[2:ks-2, 2:ks-2] = rounded_rect_mask(ks-4, ks-4, max(kr-2, 1))
        border_only = cv2.subtract(border_m, inner_m)
        key_patch[border_only > 0] = np.array(border_color, dtype=np.uint8)

        # Composite key onto overlay
        ka = (key_mask.astype(np.float32) / 255.0) * (a_val / 255.0)
        roi_b = bgr[ky:ky+ks, kx:kx+ks]
        roi_a = alpha[ky:ky+ks, kx:kx+ks].astype(np.float32) / 255.0

        for c in range(3):
            roi_b[:, :, c] = np.clip(
                key_patch[:, :, c] * ka + roi_b[:, :, c] * (1.0 - ka), 0, 255
            ).astype(np.uint8)
        alpha[ky:ky+ks, kx:kx+ks] = np.clip(
            (ka + roi_a * (1.0 - ka)) * 255, 0, 255
        ).astype(np.uint8)

        # Draw W/A/S/D letter on key
        cx_k, cy_k = kx + ks // 2, ky + ks // 2
        label = key_name.upper()
        text_scale = 0.65
        text_thickness = 2
        text_size, _ = cv2.getTextSize(label, FONT, text_scale, text_thickness)
        tx = cx_k - text_size[0] // 2
        ty = cy_k + text_size[1] // 2
        cv2.putText(bgr, label, (tx, ty), FONT, text_scale, text_color, text_thickness, cv2.LINE_AA)

    return bgr, alpha


def build_sphere_overlay(active_directions, mouse_offset=(0.0, 0.0)):
    """Build mouse sphere rotation indicator.

    Highlights every arrow in active_directions (composite rotations light
    up multiple arrows simultaneously). The mouse icon is drawn at
    mouse_offset pixels from the center, allowing the caller to feed a
    smoothly interpolated offset for fluid transitions.
    """
    if active_directions is None:
        active_directions = set()
    elif isinstance(active_directions, str):
        active_directions = {active_directions}
    else:
        active_directions = set(active_directions)

    sz = SPHERE_RADIUS * 2 + 20  # canvas size with padding
    cx, cy = sz // 2, sz // 2
    r = SPHERE_RADIUS

    bgr = np.zeros((sz, sz, 3), dtype=np.uint8)
    alpha = np.zeros((sz, sz), dtype=np.uint8)

    # Glass sphere background
    cv2.circle(bgr, (cx, cy), r, SPHERE_BG, -1, cv2.LINE_AA)
    cv2.circle(alpha, (cx, cy), r, SPHERE_ALPHA, -1, cv2.LINE_AA)

    # Sphere border (slightly brighter ring)
    cv2.circle(bgr, (cx, cy), r, SPHERE_BORDER_COLOR, 2, cv2.LINE_AA)
    cv2.circle(alpha, (cx, cy), r, SPHERE_BORDER_ALPHA, 2, cv2.LINE_AA)

    # Inner subtle gradient (lighter center for glass effect)
    glow = np.zeros((sz, sz), dtype=np.float32)
    cv2.circle(glow, (cx, cy), r - 10, 1.0, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), r * 0.4)
    for c in range(3):
        bgr[:, :, c] = np.clip(
            bgr[:, :, c].astype(np.float32) + glow * 25, 0, 255
        ).astype(np.uint8)

    # Crosshair lines
    cv2.line(bgr, (cx, cy - r + 12), (cx, cy + r - 12), CROSSHAIR_COLOR, 1, cv2.LINE_AA)
    cv2.line(bgr, (cx - r + 12, cy), (cx + r - 12, cy), CROSSHAIR_COLOR, 1, cv2.LINE_AA)

    # Direction arrows at edges of sphere
    arrow_positions = {
        "up":    (cx, cy - r + 14),
        "down":  (cx, cy + r - 14),
        "left":  (cx - r + 14, cy),
        "right": (cx + r - 14, cy),
    }
    for d, (ax, ay) in arrow_positions.items():
        color = ARROW_TIP_ACTIVE if d in active_directions else ARROW_TIP_COLOR
        draw_arrow_icon(bgr, ax, ay, d, color, size=9)

    # Mouse icon at the smoothly-interpolated offset supplied by the caller.
    mouse_cx = cx + int(round(mouse_offset[0]))
    mouse_cy = cy + int(round(mouse_offset[1]))

    m_color = MOUSE_ACTIVE_COLOR if active_directions else MOUSE_COLOR
    draw_mouse_icon(bgr, alpha, mouse_cx, mouse_cy, m_color, alpha_val=230)

    return bgr, alpha


def composite(frame, overlay_bgr, overlay_alpha, x, y):
    """Alpha-composite overlay onto frame at (x, y)."""
    oh, ow = overlay_bgr.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, fw), min(y + oh, fh)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    a = overlay_alpha[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0
    roi = frame[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = np.clip(
            overlay_bgr[oy1:oy2, ox1:ox2, c] * a + roi[:, :, c] * (1.0 - a), 0, 255
        ).astype(np.uint8)


def parse_motion_sequence(motion_sequence):
    """Return list of rendered instructions (skip: entries are dropped).

    Each instruction is the raw composite string, e.g. 'w', 'w+right', 'd+up'.
    """
    out = []
    for raw in motion_sequence.split(','):
        cmd = raw.strip().lower()
        if not cmd or cmd.startswith('skip:'):
            continue
        out.append(cmd)
    return out


def split_composite(instr):
    """Split composite instruction into (wasd_active_set, arrow_active_set).

    A composite command may contain multiple WASD letters and multiple arrow
    directions, e.g. 'a+s', 'left+up', 'w+left+up'. All are returned so the
    overlay can highlight every active component and compute a resultant
    rotation direction.
    """
    wasd = set()
    arrow = set()
    for part in instr.split('+'):
        p = part.strip()
        if p in WASD_ACTIONS:
            wasd.add(p)
        elif p in ARROW_ACTIONS:
            arrow.add(p)
    return wasd, arrow


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay WASD + mouse-sphere indicators on action-mode segments.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--segments_dir', required=True,
                   help='Directory containing segment_NNN.mp4 files '
                        '(typically output/action_demo/<sample>/).')
    p.add_argument('--motion_sequence', required=True,
                   help='Same syntax as run_action.sh (e.g. "w,w+right,left,w+up,w+down"). '
                        "'skip:' entries are dropped.")
    p.add_argument('--output', default=None,
                   help='Output mp4 path. Defaults to <segments_dir>/final_indicator.mp4.')
    p.add_argument('--target_frames', type=int, default=0,
                   help='Truncate the output to this many frames. 0 = no truncation.')
    p.add_argument('--fps', type=int, default=FPS)
    return p.parse_args()


def main():
    import imageio
    args = parse_args()
    output_path = args.output or os.path.join(args.segments_dir, 'final_indicator.mp4')

    # Discover segment files (sorted).
    seg_files = sorted(glob.glob(os.path.join(args.segments_dir, 'segment_*.mp4')))
    if not seg_files:
        raise FileNotFoundError(
            f"No segment_*.mp4 found under {args.segments_dir}")

    # Pair each segment with its (rendered) instruction.
    instructions = parse_motion_sequence(args.motion_sequence)
    if len(instructions) != len(seg_files):
        print(f"WARNING: {len(seg_files)} segment files but "
              f"{len(instructions)} rendered instructions. Pairing by index "
              f"and truncating to min.")
    n = min(len(seg_files), len(instructions))
    pairs = list(zip(seg_files[:n], instructions[:n]))

    # Pre-compute overlay dimensions
    kb_w = 3 * KEY_SIZE + 2 * KEY_GAP + 8
    kb_h = 2 * KEY_SIZE + KEY_GAP + 8
    sphere_sz = SPHERE_RADIUS * 2 + 20

    print(f"Reading {n} segments from {args.segments_dir}/...")
    frames_written = 0
    writer = None

    # Persistent mouse offset state for smooth transitions across segments.
    mouse_off_x = 0.0
    mouse_off_y = 0.0
    prev_arrow_active = set()

    for seg_idx, (seg_path, instr) in enumerate(pairs):
        cap = cv2.VideoCapture(seg_path)
        if not cap.isOpened():
            print(f"  WARNING: cannot open {seg_path}"); continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.',
                        exist_ok=True)
            writer = imageio.get_writer(
                output_path, fps=args.fps, codec='libx264', quality=8,
            )
            print(f"  Output: {width}x{height} @ {args.fps}fps -> {output_path}")
            if args.target_frames:
                print(f"  target_frames={args.target_frames}")

        # Decode active indicators for this segment (composite splits into sets
        # of active WASD keys and arrow directions).
        wasd_active, arrow_active = split_composite(instr)
        target_off_x, target_off_y = target_mouse_offset(arrow_active)

        # If the rotation direction changed from the previous segment, snap
        # the mouse instantly back to center (mimics releasing the mouse and
        # re-grabbing for the next swipe). Same-direction segments keep the
        # current offset so no unnecessary bounce happens.
        if seg_idx > 0 and arrow_active != prev_arrow_active:
            mouse_off_x = 0.0
            mouse_off_y = 0.0

        # Skip the first frame of each subsequent segment (overlap with
        # previous segment's last frame in I2V chaining).
        skip_first = (seg_idx > 0)
        frame_idx_in_seg = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip_first and frame_idx_in_seg == 0:
                frame_idx_in_seg += 1
                continue
            frame_idx_in_seg += 1

            if args.target_frames and frames_written >= args.target_frames:
                break

            # Exponential smoothing of the mouse offset toward this segments
            # target. State persists across segments so hard transitions look
            # fluid rather than snapping.
            mouse_off_x += (target_off_x - mouse_off_x) * MOUSE_SMOOTH_ALPHA
            mouse_off_y += (target_off_y - mouse_off_y) * MOUSE_SMOOTH_ALPHA

            h, w = frame.shape[:2]

            # Left: keyboard overlay (all active WASD keys highlighted)
            kb_bgr, kb_alpha = build_keyboard_overlay(wasd_active)
            kb_x = MARGIN_X
            kb_y = h - MARGIN_BOTTOM - kb_h
            composite(frame, kb_bgr, kb_alpha, kb_x, kb_y)

            # Right: sphere overlay (all active arrows highlighted, mouse eased)
            sp_bgr, sp_alpha = build_sphere_overlay(
                arrow_active, mouse_offset=(mouse_off_x, mouse_off_y))
            sp_x = w - MARGIN_X - sphere_sz
            sp_y = h - MARGIN_BOTTOM - sphere_sz
            composite(frame, sp_bgr, sp_alpha, sp_x, sp_y)

            # imageio expects RGB; OpenCV gives BGR
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_written += 1

            if frames_written % 100 == 0:
                tag = (f"{frames_written:>4}/{args.target_frames}"
                       if args.target_frames else f"{frames_written}")
                print(f"  Frame {tag}  instr={instr}")

        cap.release()
        prev_arrow_active = arrow_active
        if args.target_frames and frames_written >= args.target_frames:
            break

    if writer:
        writer.close()
    print(f"Done! {frames_written} frames -> {output_path}")


if __name__ == "__main__":
    main()
