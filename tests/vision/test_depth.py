"""Step 6: depth_left from an aligned depth map, on synthetic --depth shelves."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from shelfpulse.contracts import UNKNOWN_SKU, load_sku_master
from shelfpulse.perception import depth as D
from shelfpulse.perception.analyze import analyze, load_pipeline
from shelfpulse.perception.identify import PlanHints
from shelfpulse.perception.run import run
from shelfpulse.perception.shelves import find_shelves
from tools.eval_readings import evaluate, load
from tools.synth.make import PX_PER_CM, load_geometry, make

SKUS = load_sku_master()
S = D.DepthSettings(shelf_depth_cm=45.0)
GEO = load_geometry()
T = datetime(2026, 10, 9, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


class PlannedNames:
    """Stand-in identifier: names each pack by the demo planogram (no model needed)."""

    hints = PlanHints(["contracts/fixtures/demo_store/planograms"])

    def candidates(self, crops):
        return [None] * len(crops)

    def decide(self, cands, bay_id, row, x_cm, w_cm, ocr_words=None):
        return self.hints.planned(bay_id, row, x_cm + w_cm / 2) or UNKNOWN_SKU, 0.9


@pytest.fixture(scope="module")
def shelves_with_depth(tmp_path_factory):
    out = tmp_path_factory.mktemp("depth")
    return out, make(out, n=6, seed=5, depth=True, gaps=True, occlude=True)


def _depth_map(out: Path, frame_ref: str) -> np.ndarray:
    return cv2.imread(str(out / "depth" / f"{Path(frame_ref).stem}.png"), cv2.IMREAD_UNCHANGED)


def test_depth_left_formula():
    # 45 cm shelf, 6 cm packs: front pack 33 cm back -> 2 left; at the edge -> 7 (the most that fit)
    assert D.depth_left(33.0, 6.0, S) == 2
    assert D.depth_left(0.0, 6.0, S) == 7
    assert D.depth_left(44.9, 6.0, S) == 1  # a visible pack is at least one
    assert D.depth_left(None, 6.0, S) is None
    assert D.depth_left(10.0, None, S) is None


def test_pack_depth_needs_a_known_sku():
    assert D.pack_depth_cm("RICE_1KG", SKUS) == 6.0
    assert D.pack_depth_cm(UNKNOWN_SKU, SKUS) is None
    assert D.pack_depth_cm("AMBIGUOUS:RICE_1KG|DAL_1KG", SKUS) == 6.0  # both 6 cm deep
    assert D.pack_depth_cm("AMBIGUOUS:RICE_1KG|RICE_5KG", SKUS) is None  # 6 vs 10 cm


@pytest.mark.parametrize("noise_mm", [0, 8])
def test_truth_boxes_read_exact_depth(shelves_with_depth, noise_mm):
    out, truths = shelves_with_depth
    rng = np.random.default_rng(1)
    p = load_pipeline(backend="classic", identify=False)
    n = exact = 0
    for t in truths:
        img = cv2.imread(str(out / t.frame_ref))
        dm = _depth_map(out, t.frame_ref).astype(np.float32)
        dm = np.clip(dm + rng.normal(0, noise_mm, dm.shape), 1, 65535).astype(np.uint16)
        rails = {s.row: s.rail for s in find_shelves(img, PX_PER_CM, p.shelves)}
        for r in t.rows:
            y1 = GEO.y_px(GEO.floor_cm(r.row))
            for pk in r.packs:
                box = (
                    round(pk.x_cm * 10),
                    y1 - round(pk.h_cm * 10),
                    round((pk.x_cm + pk.w_cm) * 10),
                    y1,
                )
                recess = D.recess_cm(dm, box, rails[r.row], p.depth)
                got = D.depth_left(recess, D.pack_depth_cm(pk.sku, SKUS), p.depth)
                n += 1
                exact += got == pk.depth_left
    assert n > 100 and exact / n >= (1.0 if noise_mm == 0 else 0.97)


def test_cli_with_depth_dir_end_to_end(shelves_with_depth, tmp_path):
    out, _ = shelves_with_depth
    pipeline = dataclasses.replace(
        load_pipeline(backend="classic", identify=False), identifier=PlannedNames()
    )
    pred = tmp_path / "pred.jsonl"
    run(out / "images", pred, None, "camera", pipeline=pipeline, depth_dir=out / "depth")
    o = evaluate(load(pred), load(out / "truth.jsonl")).overall
    assert o.depth_n > 100 and o.depth_acc >= 0.97


def test_no_depth_map_means_null_and_misaligned_map_fails(shelves_with_depth):
    out, truths = shelves_with_depth
    img = cv2.imread(str(out / truths[0].frame_ref))
    pipeline = dataclasses.replace(
        load_pipeline(backend="classic", identify=False), identifier=PlannedNames()
    )
    r = analyze(img, truths[0].bay_id, "camera", T, pipeline=pipeline)
    assert all(p.depth_left is None for row in r.rows for p in row.packs)
    with pytest.raises(ValueError, match="aligned"):
        analyze(img, truths[0].bay_id, "camera", T, pipeline=pipeline, depth_map=np.ones((10, 10)))
