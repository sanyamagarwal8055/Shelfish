"""Step 2: the synthetic generator draws what its truth says, and the truth validates."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from shelfpulse.contracts import ROWS, load_sku_master, parse_bay_reading, parse_planogram
from tools.synth.make import BACKGROUND, PX_PER_CM, layout, load_geometry, main, make

GEO = load_geometry()


def _truth(out):
    lines = (out / "truth.jsonl").read_text(encoding="utf-8").splitlines()
    return [parse_bay_reading(json.loads(s)) for s in lines]


def _px(img, row, x_cm, up_cm=5.0):
    """Pixel `up_cm` above the shelf surface of `row` at x."""
    y = GEO.y_px(GEO.floor_cm(row) + up_cm)
    return img[y, round(x_cm * PX_PER_CM)].astype(int)


def _is_background(px) -> bool:
    return np.abs(px - np.array(BACKGROUND)).max() <= 10


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_plain")
    return out, make(out, n=4, seed=1)


@pytest.fixture(scope="module")
def hard(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_hard")
    return out, make(out, n=8, seed=3, gaps=True, misplaced=True, occlude=True, depth=True)


def test_writes_image_and_valid_truth_per_frame(plain):
    out, truths = plain
    assert len(_truth(out)) == 4
    for t in truths:
        img = cv2.imread(str(out / t.frame_ref))
        assert img.shape == (2100, 1200, 3)
        assert [r.row for r in t.rows] == list(ROWS)
        assert all(p.depth_left is None for r in t.rows for p in r.packs)


def test_same_seed_same_output(tmp_path):
    a = make(tmp_path / "a", n=2, seed=7, gaps=True, misplaced=True, occlude=True)
    b = make(tmp_path / "b", n=2, seed=7, gaps=True, misplaced=True, occlude=True)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    ia = cv2.imread(str(tmp_path / "a" / a[0].frame_ref))
    ib = cv2.imread(str(tmp_path / "b" / b[0].frame_ref))
    assert np.array_equal(ia, ib)


def test_drawing_matches_truth(hard):
    # Every truth pack is drawn where the truth says; every truth gap is empty background.
    out, truths = hard
    for t in truths:
        img = cv2.imread(str(out / t.frame_ref))
        for r in t.rows:
            for p in r.packs:
                assert not _is_background(_px(img, r.row, p.x_cm + p.w_cm / 2)), (t.frame_ref, p)
            for g in r.gaps:
                assert _is_background(_px(img, r.row, g.x_cm + g.w_cm / 2)), (t.frame_ref, g)


def test_full_planogram_fills_every_facing():
    skus = load_sku_master()
    plano = parse_planogram(
        json.loads(open("contracts/fixtures/demo_store/planograms/G1-L-05.json").read())
    )
    packs = layout(plano, skus, GEO)
    assert len(packs) == sum(s.facings for row in plano.rows for s in row)
    # Atta (42 cm) is taller than a shelf, so it is cut off at the shelf above.
    assert max(p.h for p in packs) == GEO.clearance_cm


def test_options_show_up_in_truth(plain, hard):
    _, base = plain
    _, t = hard
    n_gaps = lambda rs: sum(len(r.gaps) for x in rs for r in x.rows)  # noqa: E731
    assert n_gaps(t[:4]) > n_gaps(base)
    assert all(any(r.occluded for r in x.rows) for x in t)
    assert all(p.depth_left >= 1 for x in t for r in x.rows for p in r.packs)
    skus = load_sku_master()
    planned = {b: s for b, s in _planned_skus().items()}
    misplaced = [
        p for x in t for r in x.rows for p in r.packs if p.sku not in planned[x.bay_id][r.row]
    ]
    assert misplaced and all(p.sku in skus for p in misplaced)


def _planned_skus():
    out = {}
    for b in ["G1-L-04", "G1-L-05", "G3-L-05", "G7-R-02"]:
        plano = parse_planogram(
            json.loads(open(f"contracts/fixtures/demo_store/planograms/{b}.json").read())
        )
        out[b] = {i: {s.sku_id for s in row} for i, row in enumerate(plano.rows)}
    return out


def test_occluded_rows_hide_packs_and_gaps(hard):
    _, truths = hard
    for t in truths:
        for r in t.rows:
            for a, b in r.occluded:
                for p in r.packs:
                    assert min(b, p.x_cm + p.w_cm) - max(a, p.x_cm) <= p.w_cm / 2
                for g in r.gaps:
                    assert g.x_cm >= b or g.x_cm + g.w_cm <= a


def test_depth_maps_written(hard):
    out, truths = hard
    d = cv2.imread(
        str(out / "depth" / truths[0].frame_ref.split("/")[-1].replace(".jpg", ".png")),
        cv2.IMREAD_UNCHANGED,
    )
    assert d.dtype == np.uint16 and d.shape == (2100, 1200)


def test_distort_changes_image_not_truth(tmp_path):
    a = make(tmp_path / "a", n=1, seed=5)
    b = make(tmp_path / "b", n=1, seed=5, distort=True)
    assert a[0].model_dump() == b[0].model_dump()
    ia = cv2.imread(str(tmp_path / "a" / a[0].frame_ref))
    ib = cv2.imread(str(tmp_path / "b" / b[0].frame_ref))
    assert not np.array_equal(ia, ib)


def test_cli(tmp_path):
    assert main(["--n", "2", "--seed", "1", "--out", str(tmp_path), "--bays", "G3-L-05"]) == 0
    assert {t.bay_id for t in _truth(tmp_path)} == {"G3-L-05"}
    assert main(["--n", "1", "--out", str(tmp_path), "--bays", "G9-E-F"]) == 0  # random planogram
    assert {t.bay_id for t in _truth(tmp_path)} == {"G9-E-F"}
    assert main(["--n", "1", "--out", str(tmp_path), "--bays", "G99-L-01"]) == 1


@pytest.mark.uses_gallery
def test_gallery_photos_are_pasted(tmp_path):
    gallery = tmp_path / "gallery" / "SOAP_100G"
    gallery.mkdir(parents=True)
    photo = np.zeros((60, 90, 3), np.uint8)
    photo[:, :, 1] = 255  # pure green, unlike any drawn SKU colour or the background
    cv2.imwrite(str(gallery / "a.jpg"), photo)
    truths = make(tmp_path / "out", n=1, seed=1, bays=["G7-R-02"], gallery=tmp_path / "gallery")
    img = cv2.imread(str(tmp_path / "out" / truths[0].frame_ref))
    soap = next(p for r in truths[0].rows for p in r.packs if p.sku == "SOAP_100G")
    b, g, r = _px(img, 0, soap.x_cm + soap.w_cm / 2, up_cm=3.0)
    assert g > 200 and b < 60 and r < 60
