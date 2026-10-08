"""Lens correction and the warp from a camera view to a front-on bay image at 10 px/cm.

Two lens models:
- `undistort_radial(img, k1)`: one-coefficient radial model, the exact inverse of the synthetic
  `--distort` barrel in tools/synth (centre of image, radius normalised by the half-diagonal).
- `undistort_camera(img, K, dist)`: OpenCV's calibrated camera model, for real cameras.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from shelfpulse.contracts import BAY_WIDTH_CM

BAY_HEIGHT_CM = 210.0
PX_PER_CM = 10.0


def _radial_maps(h: int, w: int, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """For each undistorted pixel q, the distorted pixel p with p * (1 + k1 |p|^2) = q."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    s = math.hypot(cx, cy)
    yy, xx = np.indices((h, w), dtype=np.float32)
    qx, qy = (xx - cx) / s, (yy - cy) / s
    rq = np.sqrt(qx * qx + qy * qy)
    rp = rq.copy()
    for _ in range(20):  # fixed point: r_p = r_q / (1 + k1 r_p^2); converges for small k1
        rp = rq / (1 + k1 * rp * rp)
    scale = np.divide(rp, rq, out=np.ones_like(rq), where=rq > 0)
    return (cx + qx * scale * s).astype(np.float32), (cy + qy * scale * s).astype(np.float32)


def undistort_radial(img: np.ndarray, k1: float, nearest: bool = False) -> np.ndarray:
    if k1 == 0:
        return img
    map_x, map_y = _radial_maps(img.shape[0], img.shape[1], k1)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(img, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT)


def undistort_camera(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return cv2.undistort(img, K, dist)


def bay_size_px(px_per_cm: float = PX_PER_CM) -> tuple[int, int]:
    """(width, height) of a front-on bay image."""
    return round(BAY_WIDTH_CM * px_per_cm), round(BAY_HEIGHT_CM * px_per_cm)


def warp_to_bay(
    img: np.ndarray, corners: np.ndarray | list, px_per_cm: float = PX_PER_CM
) -> np.ndarray:
    """Warp the bay seen at `corners` (top-left, top-right, bottom-right, bottom-left, in px)
    to a front-on image: 1200 x 2100 px at 10 px/cm."""
    w, h = bay_size_px(px_per_cm)
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
