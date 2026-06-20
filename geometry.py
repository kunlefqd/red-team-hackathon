"""Camera geometry: pixel coordinates -> angle offsets."""

import math
from typing import Tuple

# From sf_robot.jsonc — the FPV camera spec.
FRAME_W = 640
FRAME_H = 480
HFOV_DEG = 120.0
VFOV_DEG = HFOV_DEG * (FRAME_H / FRAME_W)   # ~90°, close enough for steering


def pixel_to_angles(x_px: int, y_px: int) -> Tuple[float, float]:
    """Convert a pixel coordinate in the FPV frame to (yaw, pitch) offsets in degrees.

    Returns (yaw_deg, pitch_deg) relative to the camera's optical axis:
      - yaw_deg > 0  -> target is to the RIGHT of center; drone should yaw right.
      - yaw_deg < 0  -> target is to the LEFT;  drone should yaw left.
      - pitch_deg > 0 -> target is BELOW center.
      - pitch_deg < 0 -> target is ABOVE center.

    This treats the camera as a flat pinhole, which is wrong at the edges of a
    120° lens (real lenses have barrel distortion). Good enough for closed-loop
    steering: small errors get corrected on the next perception tick.
    """
    dx = x_px - FRAME_W / 2
    dy = y_px - FRAME_H / 2
    yaw_deg = (dx / (FRAME_W / 2)) * (HFOV_DEG / 2)
    pitch_deg = (dy / (FRAME_H / 2)) * (VFOV_DEG / 2)
    return yaw_deg, pitch_deg