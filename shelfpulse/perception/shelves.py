"""Shelf rows from shelf-rail detection, and occlusion from covered rails.

A rail is a horizontal band 1.5-4.5 cm tall whose top and bottom edges run across most of the
image. Each rail's top edge is a shelf surface; the row is the strip above it, up to the next rail.
Row numbers come from height above the bay bottom (row r sits at r * pitch), so one missed rail
doesn't renumber the rows above it.

Anything standing in front of a rail (a person, a trolley) breaks its colour: those columns are
the row's occluded x ranges.
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
    out = []
    idx = np.flatnonzero(off)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            a, b = run[0] / px_per_cm, (run[-1] + 1) / px_per_cm
            if b - a >= s.min_occlusion_cm:
                out.append((round(float(a), 1), round(float(b), 1)))
    return out


def find_shelves(
    img: np.ndarray, px_per_cm: float, s: ShelfSettings = DEFAULT_SHELVES
) -> list[Shelf]:
    """One Shelf per detected row, bottom row first."""
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
        occ = _occluded(img, rail, px_per_cm, s)
        shelves.append(Shelf(r, top, rail.y0, rail, occ))
    return shelves
