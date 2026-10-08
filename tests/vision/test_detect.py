"""Step 4: rectify, shelf rails, occlusion and the classic detector, scored on synthetic shelves."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from shelfpulse.config import load_yaml
from shelfpulse.contracts import UNKNOWN_SKU
from shelfpulse.perception.analyze import default_pipeline
from shelfpulse.perception.detector import Detector
from shelfpulse.perception.rectify import undistort_radial, warp_to_bay
from shelfpulse.perception.run import bay_from_name, run
from shelfpulse.perception.shelves import find_shelves
from tools.eval_readings import evaluate, load
from tools.synth.make import RAIL_CM, barrel, load_geometry, make

GEO = load_geometry()
SYNTH_K1 = load_yaml("perception")["rectify"]["synth_k1"]


def _score(tmp_path: Path, name: str, undistort: bool = False, **opts):
    gold = tmp_path / name
    make(gold, n=8, seed=11, **opts)
    pred = tmp_path / f"{name}_pred.jsonl"
    k1 = SYNTH_K1 if undistort else 0.0
    run(gold / "images", pred, None, "camera", undistort_k1=k1)
    return evaluate(load(pred), load(gold / "truth.jsonl")).overall, load(pred)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("detect")
    make(out / "plain", n=4, seed=3)
    make(out / "occ", n=4, seed=3, occlude=True)
    return out


def _img(folder: Path, i: int = 0) -> np.ndarray:
    return cv2.imread(str(sorted((folder / "images").glob("*.jpg"))[i]))


# --- rectify -----------------------------------------------------------------------------------


def test_undistort_inverts_synthetic_barrel(synth):
    img = _img(synth / "plain")
    back = undistort_radial(barrel(img, SYNTH_K1), SYNTH_K1)
    c = (slice(100, -100), slice(100, -100))
    assert np.abs(back[c].astype(int) - img[c].astype(int)).mean() < 6
    assert undistort_radial(img, 0.0) is img


def test_warp_to_bay_recovers_front_on_view(synth):
    img = _img(synth / "plain")
    corners = np.float32([[300, 200], [1500, 260], [1450, 2300], [330, 2380]])
    src = np.float32([[0, 0], [1199, 0], [1199, 2099], [0, 2099]])
    view = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, corners), (1800, 2600))
    back = warp_to_bay(view, corners)
    assert back.shape == img.shape
    c = (slice(50, -50), slice(50, -50))
    assert np.abs(back[c].astype(int) - img[c].astype(int)).mean() < 8


# --- shelves ------------------------------------------------------------------------------------


def test_finds_six_rows_at_the_right_heights(synth):
    shelves = find_shelves(_img(synth / "plain"), 10.0, default_pipeline().shelves)
    assert [s.row for s in shelves] == [0, 1, 2, 3, 4, 5]
    for s in shelves:
        assert abs(s.surface - GEO.y_px(s.row * GEO.pitch_cm + RAIL_CM)) <= 3
    assert all(not s.occluded for s in shelves)


def test_person_in_front_marks_rows_occluded(synth):
    shelves = find_shelves(_img(synth / "occ"), 10.0, default_pipeline().shelves)
    assert any(s.occluded for s in shelves)
    assert shelves[0].occluded  # a standing person always covers the bottom shelf


# --- detector -----------------------------------------------------------------------------------


def test_clean_shelves(tmp_path):
    o, preds = _score(tmp_path, "plain")
    assert o.facing_err <= 0.05 and o.gap_recall == 1.0
    assert all(p.sku == UNKNOWN_SKU for r in preds for row in r.rows for p in row.packs)
    assert all(r.quality >= 0.9 for r in preds)


def test_gaps_found(tmp_path):
    o, _ = _score(tmp_path, "gaps", gaps=True)
    assert o.gap_recall >= 0.95 and o.facing_err <= 0.1


def test_occluded_shelves(tmp_path):
    o, preds = _score(tmp_path, "occ", gaps=True, occlude=True)
    assert o.facing_err <= 0.6 and o.gap_recall >= 0.9
    assert all(r.quality < 0.95 for r in preds)  # a person in view lowers the quality


def test_undistort_helps_on_distorted_shelves(tmp_path):
    raw, _ = _score(tmp_path, "dist_raw", gaps=True, distort=True)
    fixed, _ = _score(tmp_path, "dist", undistort=True, gaps=True, distort=True)
    assert fixed.facing_err < raw.facing_err and fixed.gap_recall > raw.gap_recall


def test_bay_id_from_file_name():
    assert bay_from_name("runs/synth01/images/0003_G7-R-02.jpg") == "G7-R-02"
    assert bay_from_name("photos/IMG_001.jpg") is None


def test_bad_backend_and_missing_weights_fail_clearly():
    with pytest.raises(ValueError):
        Detector("resnet")
    with pytest.raises(FileNotFoundError, match="train_sku110k"):
        Detector("yolo", weights="data/models/does_not_exist.pt")


WEIGHTS = Path(load_yaml("perception")["detector"]["weights"])


@pytest.mark.skipif(not WEIGHTS.is_file(), reason=f"no YOLO weights at {WEIGHTS}")
def test_yolo_backend_runs_on_cpu(synth):
    from shelfpulse.perception.analyze import load_pipeline

    p = load_pipeline(backend="yolo")
    img = _img(synth / "plain")
    boxes = p.detector.detect(img, find_shelves(img, 10.0, p.shelves), 10.0)
    assert all(0 <= b.x0 < b.x1 <= img.shape[1] for b in boxes)
