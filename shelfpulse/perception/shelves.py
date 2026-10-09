"""Shelf rows from shelf-rail detection, and occlusion from covered rails.

A rail is a horizontal band 1.5-4.5 cm tall whose top and bottom edges run across most of the
image. Each rail's top edge is a shelf surface; the row is the strip above it, up to the next rail.
Row numbers come from height above the bay bottom (row r sits at r * pitch), so one missed rail
doesn't renumber the rows above it.

Occluded x ranges (a person, a trolley in front of the shelf): with an aligned depth map, columns
of the row where enough pixels sit closer than the shelf edge (the rail's depth); without one,
columns where something breaks the rail's colour. Depth is preferred: shelf-edge price labels
change the rail's colour too, but not its depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shelfpulse.contracts import ROWS


@dataclass(frozen=True)
class ShelfSettings:
    pitch_cm: float = 35.0  # bay height / levels
    rail_cm: float = 3.0
    rail_band_cm: tuple[float, float] = (1.5, 4.5)
    edge_thr: float = 40.0  # Sobel |dI/dy| for a strong horizontal edge
    min_cover: float = 0.4  # fraction of columns an edge must span
    occlusion_tol: float = 45.0  # colour distance from the rail's own colour
    min_occlusion_cm: float = 5.0
    depth_margin_mm: float = 50.0  # closer than the shelf edge by this much = in front of it
    depth_cover: float = 0.05  # share of a column's row pixels in front of the edge to count


DEFAULT_SHELVES = ShelfSettings()


@dataclass(frozen=True)
class Rail:
    y0: int  # top edge = shelf surface (px)
    y1: int  # bottom edge (px)
    cover: float


@dataclass(frozen=True)
class Shelf:
    row: int
    top: int  # px: bottom of the rail above (or 0)
    surface: int  # px: top of this row's rail
    rail: Rail
    occluded: list[tuple[float, float]]  # x ranges in cm


def _edge_rows(cover: np.ndarray, min_cover: float) -> list[int]:
    """Centres of runs of image rows whose edge coverage is at least min_cover."""
    idx = np.flatnonzero(cover >= min_cover)
    if idx.size == 0:
        return []
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [int(round(r.mean())) for r in runs]


def find_rails(img: np.ndarray, px_per_cm: float, s: ShelfSettings = DEFAULT_SHELVES) -> list[Rail]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = gray.shape[0]
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    valid = gray > 8  # black borders after undistort don't count
    cover = ((gy > s.edge_thr) & valid).sum(1) / np.maximum(valid.sum(1), 1)
    edges = _edge_rows(cover, s.min_cover)
    # The bay's bottom rail ends at the image edge, which has no gradient: add it as an edge.
    if not edges or edges[-1] < h - 2:
        edges.append(h - 1)

    # Every edge pair 1.5-4.5 cm apart with no full-width edge between them is a candidate rail.
    # Candidates may overlap (a pack's dark bottom edge sits just above the real rail);
    # find_shelves keeps the one nearest each row's expected height.
    lo, hi = (round(c * px_per_cm) for c in s.rail_band_cm)
    return [
        Rail(a, b, float(cover[max(a - 1, 0) : a + 2].max()))
        for a, b in zip(edges, edges[1:], strict=False)  # neighbours: nothing strong in between
        if lo <= b - a <= hi
    ]


def _occluded(img: np.ndarray, rail: Rail, px_per_cm: float, s: ShelfSettings) -> list:
    band = img[rail.y0 + 2 : max(rail.y1 - 1, rail.y0 + 3)].astype(np.float32)
    if band.size == 0:
        return []
    col = np.median(band, axis=0)  # one colour per column
    ref = np.median(col, axis=0)
    off = np.abs(col - ref).max(1) > s.occlusion_tol
    return _ranges(off, px_per_cm, s.min_occlusion_cm)


def _ranges(mask: np.ndarray, px_per_cm: float, min_cm: float) -> list[tuple[float, float]]:
    """Column mask -> x ranges in cm at least min_cm wide."""
    out = []
    idx = np.flatnonzero(mask)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            a, b = run[0] / px_per_cm, (run[-1] + 1) / px_per_cm
            if b - a >= min_cm:
                out.append((round(float(a), 1), round(float(b), 1)))
    return out


EDGE_CLUSTER = 30.0  # depth units (mm): readings this close to the far end are the rail itself


def shelf_edge_depth(depth: np.ndarray, rail: Rail) -> float | None:
    """Distance to the shelf edge along a rail (None if no readings).

    People and trolleys only ever stand in front of a rail, never behind it, and may hide most
    of it; so the edge is the far end of the rail band's readings (90th percentile), refined to
    the median of readings within EDGE_CLUSTER of it. Works while >= ~10% of the rail shows.
    """
    band = depth[rail.y0 + 1 : max(rail.y1 - 1, rail.y0 + 2)]
    vals = band[band > 0].astype(np.float32)
    if not vals.size:
        return None
    far = float(np.percentile(vals, 90))
    return float(np.median(vals[np.abs(vals - far) <= EDGE_CLUSTER]))


def _occluded_depth(
    depth: np.ndarray, top: int, rail: Rail, px_per_cm: float, s: ShelfSettings
) -> list[tuple[float, float]]:
    edge = shelf_edge_depth(depth, rail)
    if edge is None:
        return []
    region = depth[top : rail.y1].astype(np.float32)
    front = (region > 0) & (region < edge - s.depth_margin_mm)
    return _ranges(front.mean(0) >= s.depth_cover, px_per_cm, s.min_occlusion_cm)


def find_shelves(
    img: np.ndarray,
    px_per_cm: float,
    s: ShelfSettings = DEFAULT_SHELVES,
    depth: np.ndarray | None = None,
) -> list[Shelf]:
    """One Shelf per detected row, bottom row first. `depth`: aligned depth map (optional)."""
    h = img.shape[0]
    rails = sorted(find_rails(img, px_per_cm, s), key=lambda r: r.y0, reverse=True)
    # A row of short, aligned packs can look like a rail too; per row keep the candidate nearest
    # the row's expected height.
    by_row: dict[int, tuple[float, Rail]] = {}
    for rail in rails:
        height_cm = (h - rail.y0) / px_per_cm - s.rail_cm  # surface height above the bay bottom
        r = round(height_cm / s.pitch_cm)
        off = abs(height_cm - r * s.pitch_cm)
        if r in ROWS and off < s.pitch_cm / 3:
            if r not in by_row or (off, -rail.cover) < (by_row[r][0], -by_row[r][1].cover):
                by_row[r] = (off, rail)
    chosen = [rail for _, rail in by_row.values()]
    shelves = []
    for r in sorted(by_row):
        rail = by_row[r][1]
        above = [x for x in chosen if x.y1 <= rail.y0]
        top = max((x.y1 for x in above), default=0)
        if depth is not None:
            occ = _occluded_depth(depth, top, rail, px_per_cm, s)
        else:
            occ = _occluded(img, rail, px_per_cm, s)
        shelves.append(Shelf(r, top, rail.y0, rail, occ))
    return shelves
