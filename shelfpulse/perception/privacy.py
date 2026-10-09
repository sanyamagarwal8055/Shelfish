"""People in front of the shelf: find them, measure how much of a bay they hide, blur them.

With an aligned depth map (stereo cameras and the robot's ToF both give one), anything closer than
the shelf edge by more than a margin is a person, trolley or other obstacle: nothing on a shelf
stands in front of its own edge. No face detection and no identity tracking; masked pixels are
blurred before any frame is saved.
"""

from __future__ import annotations

import cv2
import numpy as np

from shelfpulse.perception.shelves import Shelf, shelf_edge_depth


def people_mask(
    depth: np.ndarray | None, shelves: list[Shelf], margin_mm: float = 50.0
) -> np.ndarray | None:
    """Pixels in front of the shelf edge, or None when it can't be told (no depth / no rails)."""
    if depth is None:
        return None
    edges = [e for s in shelves if (e := shelf_edge_depth(depth, s.rail)) is not None]
    if not edges:
        return None
    edge = float(np.median(edges))
    return (depth > 0) & (depth < edge - margin_mm)


def person_cover(mask: np.ndarray | None, shelves: list[Shelf]) -> float:
    """Share of the bay hidden by people. No shelves found at all counts as fully hidden."""
    if not shelves:
        return 1.0
    if mask is None:
        return 0.0
    return float(mask.mean())


def blur_people(img: np.ndarray, mask: np.ndarray | None, px_per_cm: float = 10.0) -> np.ndarray:
    """A copy with masked pixels (grown by ~3 cm) pixelated into ~8 cm blocks (fast, unreadable)."""
    if mask is None or not mask.any():
        return img.copy()
    grow = max(3, round(3 * px_per_cm)) | 1
    m = cv2.dilate(mask.astype(np.uint8), np.ones((grow, grow), np.uint8)).astype(bool)
    h, w = img.shape[:2]
    block = max(4, round(8 * px_per_cm))
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    out[m] = coarse[m]
    return out
