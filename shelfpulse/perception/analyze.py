"""analyze(bay_image) -> BayReading: the single entry point from pixels to the contract.

The image must already be a front-on bay (rectify.py does that upstream). Steps:
shelf rails -> rows + occlusion, detector -> pack boxes, boxes -> packs per row, gaps = row
stretches wider than 5 cm with no packs, quality score. sku stays "UNKNOWN" until identify (step 5)
and depth_left stays null until depth (step 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import numpy as np

from shelfpulse.config import load_yaml
from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    MIN_GAP_CM,
    UNKNOWN_SKU,
    BayReading,
    Gap,
    Pack,
    Row,
    Source,
)
from shelfpulse.perception import quality
from shelfpulse.perception.detector import Box, ClassicSettings, Detector
from shelfpulse.perception.shelves import Shelf, ShelfSettings, find_shelves


@dataclass(frozen=True)
class Pipeline:
    shelves: ShelfSettings
    detector: Detector
    row_tolerance_cm: float = 8.0
    sharp_ref_var: float = 100.0


def load_pipeline(name: str = "perception", backend: str | None = None) -> Pipeline:
    """Build the pipeline from configs/perception.yaml; `backend` overrides detector.backend."""
    cfg = load_yaml(name)
    bay = load_yaml("store_layout")["bay"]
    sh = cfg["shelves"]
    shelves = ShelfSettings(
        pitch_cm=float(bay["height_m"]) * 100 / int(bay["levels"]),
        rail_cm=sh["rail_cm"],
        rail_band_cm=tuple(sh["rail_band_cm"]),
        edge_thr=sh["edge_thr"],
        min_cover=sh["min_cover"],
        occlusion_tol=sh["occlusion_tol"],
        min_occlusion_cm=sh["min_occlusion_cm"],
    )
    d = cfg["detector"]
    c = d["classic"]
    classic = ClassicSettings(
        bg_tol=c["bg_tol"],
        band_cm=tuple(c["band_cm"]),
        min_pack_cm=c["min_pack_cm"],
        edge_thr=c["edge_thr"],
        edge_cover=c["edge_cover"],
        conf=c["conf"],
    )
    detector = Detector(backend or d["backend"], d["weights"], d["conf"], d["imgsz"], classic)
    return Pipeline(shelves, detector, d["row_tolerance_cm"], cfg["quality"]["sharp_ref_var"])


@lru_cache(maxsize=1)
def default_pipeline() -> Pipeline:
    return load_pipeline()


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def free_stretches(taken: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of [0, 120] cm not covered by `taken`."""
    out, x = [], 0.0
    for a, b in sorted(taken):
        if a > x:
            out.append((x, a))
        x = max(x, b)
    if x < BAY_WIDTH_CM:
        out.append((x, BAY_WIDTH_CM))
    return out


def build_rows(
    shelves: list[Shelf], boxes: list[Box], px_per_cm: float, row_tolerance_cm: float = 8.0
) -> list[Row]:
    """Assign each box to the shelf it stands on, then derive gaps per row."""
    tol = row_tolerance_cm * px_per_cm
    per_row: dict[int, list[Box]] = {s.row: [] for s in shelves}
    for b in boxes:
        best = min(shelves, key=lambda s: abs(b.y1 - s.surface), default=None)
        if best is not None and abs(b.y1 - best.surface) <= tol:
            per_row[best.row].append(b)

    rows = []
    for s in shelves:
        packs = []
        for b in sorted(per_row[s.row], key=lambda b: b.x0):
            x0 = min(max(b.x0 / px_per_cm, 0.0), BAY_WIDTH_CM)
            x1 = min(max(b.x1 / px_per_cm, 0.0), BAY_WIDTH_CM)
            h = (b.y1 - b.y0) / px_per_cm
            # Like the truth: a pack more than half hidden isn't reported.
            if x1 - x0 <= 0 or h <= 0:
                continue
            if any(_overlap(x0, x1, a, z) > (x1 - x0) / 2 for a, z in s.occluded):
                continue
            packs.append((x0, x1, h, b.conf))
        taken = [(x0, x1) for x0, x1, _, _ in packs] + list(s.occluded)
        stretches = [(round(a, 2), round(b, 2)) for a, b in free_stretches(taken)]
        rows.append(
            Row(
                row=s.row,
                occluded=[[a, b] for a, b in s.occluded],
                packs=[
                    Pack(
                        sku=UNKNOWN_SKU,
                        conf=round(min(max(conf, 0.0), 1.0), 3),
                        x_cm=round(x0, 2),
                        w_cm=round(x1 - x0, 2),
                        h_cm=round(h, 2),
                        stack=1,
                        depth_left=None,
                    )
                    for x0, x1, h, conf in packs
                ],
                gaps=[
                    Gap(x_cm=a, w_cm=round(b - a, 2))
                    for a, b in stretches
                    if round(b - a, 2) > MIN_GAP_CM
                ],
            )
        )
    return rows


def analyze(
    bay_image: np.ndarray,
    bay_id: str,
    source: Source,
    t: datetime,
    frame_ref: str = "",
    pipeline: Pipeline | None = None,
) -> BayReading:
    """Read one front-on bay image (H x W x 3, BGR) into a BayReading.

    `t` must be timezone-aware. Raises pydantic.ValidationError if the result breaks the contract.
    """
    if bay_image.ndim != 3 or bay_image.shape[1] == 0:
        raise ValueError(f"bay_image must be H x W x 3, got shape {bay_image.shape}")
    p = pipeline or default_pipeline()
    px_per_cm = bay_image.shape[1] / BAY_WIDTH_CM
    shelves = find_shelves(bay_image, px_per_cm, p.shelves)
    boxes = p.detector.detect(bay_image, shelves, px_per_cm)
    return BayReading(
        bay_id=bay_id,
        source=source,
        t=t,
        frame_ref=frame_ref,
        quality=quality.score(bay_image, shelves, p.sharp_ref_var),
        px_per_cm=px_per_cm,
        rows=build_rows(shelves, boxes, px_per_cm, p.row_tolerance_cm),
    )
