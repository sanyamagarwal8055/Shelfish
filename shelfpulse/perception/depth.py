"""How many packs are left behind each front pack (depth_left), from an aligned depth map.

The depth map is aligned pixel-for-pixel with the front-on bay image and holds distance from the
camera (mm by default; 0 = no reading), like a stereo or ToF camera. Each pack's recess is
measured against its own shelf's rail, so the camera's distance and tilt drop out:

    recess_cm  = (median depth of the pack's front face - median depth of the rail below) / 10
    depth_left = ceil((shelf_depth - recess) / pack_depth_cm)       (step 6 of the Vision role)

Shoppers take from the front, so the remaining packs sit at the back; a pack pushed 30 cm into a
45 cm shelf with 6 cm packs means 2-3 left. pack_depth_cm comes from data/sku_master.csv, so the
SKU must be known: UNKNOWN packs (and AMBIGUOUS ones whose candidates differ in depth) get null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shelfpulse.contracts import UNKNOWN_SKU, SkuRow, ambiguous_candidates
from shelfpulse.perception.shelves import Rail, shelf_edge_depth


@dataclass(frozen=True)
class DepthSettings:
    shelf_depth_cm: float = 45.0  # configs/store_layout.yaml bay.depth_cm
    mm_per_unit: float = 1.0  # depth map units
    round_tol: float = 0.15  # packs; absorbs measurement noise before rounding up
    min_valid_frac: float = 0.3  # share of a region with a depth reading to trust it


def _median(region: np.ndarray, min_frac: float) -> float | None:
    vals = region[region > 0]
    if region.size == 0 or vals.size < min_frac * region.size:
        return None
    return float(np.median(vals))


def pack_depth_cm(sku: str, skus: dict[str, SkuRow]) -> float | None:
    """Pack depth for a SKU reference; None if unknown or the candidates disagree."""
    if sku == UNKNOWN_SKU:
        return None
    depths = {skus[c].depth_cm for c in ambiguous_candidates(sku) if c in skus}
    return depths.pop() if len(depths) == 1 else None


def recess_cm(
    depth: np.ndarray,
    box: tuple[int, int, int, int],
    rail: Rail,
    s: DepthSettings,
) -> float | None:
    """How far the pack's front face sits behind the shelf edge, in cm (None if unreadable)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4:
        return None
    e = shelf_edge_depth(depth, rail)  # robust to people hiding most of the rail
    face = depth[y0 + h // 4 : y1 - h // 4, x0 + w // 4 : x1 - w // 4]  # centre of the front
    if e is None:
        return None
    # Nothing on the shelf stands in front of its edge: closer readings are people or noise.
    margin = 10.0 / s.mm_per_unit  # 1 cm
    f = _median(np.where(face >= e - margin, face, 0), s.min_valid_frac)
    if f is None:
        return None
    return min(max((f - e) * s.mm_per_unit / 10.0, 0.0), s.shelf_depth_cm)


def depth_left(recess: float | None, pack_depth: float | None, s: DepthSettings) -> int | None:
    if recess is None or not pack_depth:
        return None
    n = math.ceil((s.shelf_depth_cm - recess) / pack_depth - s.round_tol)
    fit = max(1, math.floor(s.shelf_depth_cm / pack_depth + s.round_tol))
    return min(max(n, 1), fit)  # a pack is in view, so at least 1; never more than fit
