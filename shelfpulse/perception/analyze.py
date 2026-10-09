"""analyze(bay_image) -> BayReading: the single entry point from pixels to the contract.

The image must already be a front-on bay (rectify.py does that upstream). Steps:
shelf rails -> rows + occlusion, detector -> pack boxes, boxes -> packs per row, gaps = row
stretches wider than 5 cm with no packs, product names from the gallery index (identify.py;
"UNKNOWN" for every pack if the index hasn't been built), depth_left from an aligned depth map
when one is given (depth.py; null otherwise), quality score.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from shelfpulse.config import REPO_ROOT, load_yaml
from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    MIN_GAP_CM,
    UNKNOWN_SKU,
    BayReading,
    Gap,
    Pack,
    Row,
    SkuRow,
    Source,
    load_sku_master,
)
from shelfpulse.perception import depth as depth_mod
from shelfpulse.perception import quality
from shelfpulse.perception.detector import Box, ClassicSettings, Detector
from shelfpulse.perception.identify import Identifier, IdentifySettings, PlanHints
from shelfpulse.perception.shelves import Shelf, ShelfSettings, find_shelves


@dataclass(frozen=True)
class Pipeline:
    shelves: ShelfSettings
    detector: Detector
    row_tolerance_cm: float = 8.0
    sharp_ref_var: float = 100.0
    identifier: Identifier | None = None  # None: every pack stays UNKNOWN
    depth: depth_mod.DepthSettings = depth_mod.DepthSettings()
    skus: dict[str, SkuRow] | None = None  # pack depths for depth_left


def load_pipeline(
    name: str = "perception", backend: str | None = None, identify: bool = True
) -> Pipeline:
    """Build the pipeline from configs/perception.yaml; `backend` overrides detector.backend.

    With identify, products are named from identify.index_dir if that index exists.
    """
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
        depth_margin_mm=sh["depth_margin_mm"],
        depth_cover=sh["depth_cover"],
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
    detector = Detector(
        _backend(backend or d["backend"], d["weights"]),
        d["weights"],
        d["conf"],
        d["imgsz"],
        classic,
        per_row=d["per_row"],
        row_margin_cm=d["row_margin_cm"],
    )
    identifier = load_identifier(name) if identify else None
    dc = cfg["depth"]
    depth = depth_mod.DepthSettings(
        shelf_depth_cm=float(bay["depth_cm"]),
        mm_per_unit=dc["mm_per_unit"],
        round_tol=dc["round_tol"],
        min_valid_frac=dc["min_valid_frac"],
    )
    return Pipeline(
        shelves,
        detector,
        d["row_tolerance_cm"],
        cfg["quality"]["sharp_ref_var"],
        identifier,
        depth,
        load_sku_master(),
    )


def _backend(backend: str, weights: str) -> str:
    """'auto': YOLO when its weights file exists, else the classic baseline (with a note)."""
    if backend != "auto":
        return backend
    path = Path(weights) if Path(weights).is_absolute() else REPO_ROOT / weights
    if path.is_file():
        return "yolo"
    print(
        f"note: no YOLO weights at {path}; using the classic detector (synthetic shelves only)",
        file=sys.stderr,
    )
    return "classic"


def load_identifier(name: str = "perception") -> Identifier | None:
    """The gallery identifier from config, or None (with a note) if its index isn't built."""
    c = load_yaml(name)["identify"]
    index_dir = Path(c["index_dir"])
    index_dir = index_dir if index_dir.is_absolute() else REPO_ROOT / index_dir
    if not (index_dir / "index.faiss").is_file():
        print(
            f"note: no gallery index at {index_dir}; packs stay UNKNOWN. "
            "Build it with: python scripts/build_gallery_index.py",
            file=sys.stderr,
        )
        return None
    from shelfpulse.perception.embedder import Embedder  # torch/transformers only when used
    from shelfpulse.perception.sku_index import SkuIndex

    index = SkuIndex.load(index_dir)
    embedder = Embedder(c["backend"], c.get("model"), c["input_px"])
    if index.model != embedder.name:
        raise ValueError(
            f"gallery index was built with {index.model}, config asks for {embedder.name}; "
            "rebuild it with scripts/build_gallery_index.py"
        )
    hints = PlanHints(c["planogram_dirs"])
    return Identifier(embedder, index, load_sku_master(), hints, load_identify_settings(name))


def load_identify_settings(name: str = "perception") -> IdentifySettings:
    c = load_yaml(name)["identify"]
    keys = (
        "top_k",
        "unknown_below",
        "ambiguous_margin",
        "location_bonus",
        "size_bonus",
        "size_tol",
    )
    return IdentifySettings(**{k: c[k] for k in keys})


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


Namer = Callable[[Box, int, float, float], tuple[str, float]]  # (box, row, x_cm, w_cm) -> sku, conf
Depther = Callable[[Box, Shelf, str], "int | None"]  # (box, shelf, sku) -> depth_left


def build_rows(
    shelves: list[Shelf],
    boxes: list[Box],
    px_per_cm: float,
    row_tolerance_cm: float = 8.0,
    namer: Namer | None = None,
    depther: Depther | None = None,
) -> list[Row]:
    """Assign each box to the shelf it stands on, name it, then derive gaps per row."""
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
            sku, conf = namer(b, s.row, x0, x1 - x0) if namer else (UNKNOWN_SKU, b.conf)
            left = depther(b, s, sku) if depther else None
            packs.append((x0, x1, h, conf, sku, left))
        taken = [(x0, x1) for x0, x1, *_ in packs] + list(s.occluded)
        stretches = [(round(a, 2), round(b, 2)) for a, b in free_stretches(taken)]
        rows.append(
            Row(
                row=s.row,
                occluded=[[a, b] for a, b in s.occluded],
                packs=[
                    Pack(
                        sku=sku,
                        conf=round(min(max(conf, 0.0), 1.0), 3),
                        x_cm=round(x0, 2),
                        w_cm=round(x1 - x0, 2),
                        h_cm=round(h, 2),
                        stack=1,
                        depth_left=left,
                    )
                    for x0, x1, h, conf, sku, left in packs
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
    depth_map: np.ndarray | None = None,
) -> BayReading:
    """Read one front-on bay image (H x W x 3, BGR) into a BayReading.

    `depth_map` (H x W, distance from the camera, 0 = no reading) is aligned with the image;
    without it depth_left is null. `t` must be timezone-aware. Raises pydantic.ValidationError
    if the result breaks the contract.
    """
    if bay_image.ndim != 3 or bay_image.shape[1] == 0:
        raise ValueError(f"bay_image must be H x W x 3, got shape {bay_image.shape}")
    if depth_map is not None and depth_map.shape[:2] != bay_image.shape[:2]:
        raise ValueError(f"depth_map {depth_map.shape} not aligned with image {bay_image.shape}")
    p = pipeline or default_pipeline()
    px_per_cm = bay_image.shape[1] / BAY_WIDTH_CM
    shelves = find_shelves(bay_image, px_per_cm, p.shelves, depth_map)
    boxes = p.detector.detect(bay_image, shelves, px_per_cm)
    namer = _namer(p.identifier, bay_image, boxes, bay_id) if p.identifier and boxes else None
    depther = _depther(depth_map, p) if depth_map is not None else None
    return BayReading(
        bay_id=bay_id,
        source=source,
        t=t,
        frame_ref=frame_ref,
        quality=quality.score(bay_image, shelves, p.sharp_ref_var),
        px_per_cm=px_per_cm,
        rows=build_rows(shelves, boxes, px_per_cm, p.row_tolerance_cm, namer, depther),
    )


def _depther(depth_map: np.ndarray, p: Pipeline) -> Depther:
    skus = p.skus or {}

    def left(b: Box, shelf: Shelf, sku: str) -> int | None:
        box = (int(b.x0), int(b.y0), int(round(b.x1)), int(round(b.y1)))
        recess = depth_mod.recess_cm(depth_map, box, shelf.rail, p.depth)
        return depth_mod.depth_left(recess, depth_mod.pack_depth_cm(sku, skus), p.depth)

    return left


def _namer(ident: Identifier, img: np.ndarray, boxes: list[Box], bay_id: str) -> Namer:
    """Embed every box crop in one batch; the namer then decides per pack with its row and x."""
    h, w = img.shape[:2]
    crops = []
    for b in boxes:
        x0, x1 = max(int(b.x0), 0), min(int(round(b.x1)), w)
        y0, y1 = max(int(b.y0), 0), min(int(round(b.y1)), h)
        crops.append(img[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)])
    cands = {id(b): c for b, c in zip(boxes, ident.candidates(crops), strict=True)}

    def name(b: Box, row: int, x_cm: float, w_cm: float) -> tuple[str, float]:
        sku, conf = ident.decide(cands[id(b)], bay_id, row, x_cm, w_cm)
        return sku, (b.conf if sku == UNKNOWN_SKU else conf)

    return name
