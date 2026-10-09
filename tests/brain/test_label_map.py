"""Step 6: label maps from robot shelf labels, drift vs the digital planogram, brain wiring."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shelfpulse import bus
from shelfpulse.brain import load_brain_config, run
from shelfpulse.contracts import load_sku_master, parse_bay_reading, parse_planogram
from shelfpulse.decision.types import SlotObservation
from shelfpulse.planogram.label_map import Drift, LabelMaps, drift_rows, rows_from_labels

ROOT = Path(__file__).resolve().parents[2]
SKUS_PATH = ROOT / "data" / "sku_master.csv"
SKUS = load_sku_master(SKUS_PATH)
DEMO = ROOT / "contracts" / "fixtures" / "demo_store" / "planograms"
CFG = load_brain_config()
IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 10, 8, 7, 5, tzinfo=IST)


def robot(bay: str, rows: dict[int, list[tuple[float, str]]], t: datetime = T0):
    return parse_bay_reading({
        "bay_id": bay, "source": "robot", "t": t.isoformat(), "frame_ref": "r.jpg",
        "quality": 0.95, "px_per_cm": 10.0,
        "rows": [{"row": r, "labels": [{"x_cm": x, "sku": s, "price": 10.0} for x, s in labels]}
                 for r, labels in rows.items()],
    })  # fmt: skip


def plan(bay: str):
    return parse_planogram(json.loads((DEMO / f"{bay}.json").read_text()))


# --- building --------------------------------------------------------------------------------


def test_each_label_owns_shelf_to_the_next():
    rows = rows_from_labels(robot("G1-L-04", {3: [(0.4, "RICE_1KG"), (60.4, "DAL_1KG")]}),
                            SKUS, CFG.label_map)  # fmt: skip
    rice, dal = rows[3]
    assert (rice.sku_id, rice.x_start_cm, rice.x_end_cm) == ("RICE_1KG", 0.0, 60.4)  # snapped
    assert (dal.x_start_cm, dal.x_end_cm, dal.position) == (60.4, 120.0, 1)
    assert rice.facings == 6 and rice.min_facings == 2  # 60.4 // 9.5 cm; ceil(6 x 0.25)


def test_unknown_label_skus_are_ignored():
    rows = rows_from_labels(robot("G1-L-04", {0: [(0.5, "NOT_A_SKU")]}), SKUS, CFG.label_map)
    assert rows == {}


def test_update_keeps_rows_not_read_this_time():
    lm = LabelMaps(SKUS, CFG.label_map)
    lm.update(robot("G6-R-06", {0: [(0.5, "DETERGENT_1KG")], 1: [(0.5, "DETERGENT_1KG")]}))
    changed = lm.update(robot("G6-R-06", {1: [(0.5, "SOAP_100G")]}, T0 + timedelta(hours=8)))
    assert [r[0].sku_id for r in changed.rows] == ["DETERGENT_1KG", "SOAP_100G"]
    assert changed.source == "label_map" and changed.updated == T0 + timedelta(hours=8)
    assert lm.update(robot("G6-R-06", {1: [(0.5, "SOAP_100G")]})) is None  # nothing changed


def test_camera_readings_are_ignored():
    lm = LabelMaps(SKUS, CFG.label_map)
    cam = robot("G1-L-04", {0: []}).model_copy(update={"source": "camera"})
    assert lm.update(cam) is None


# --- drift -----------------------------------------------------------------------------------


def labels_like_g1_l_04(row3):
    """Robot labels matching the G1-L-04 planogram, with row 3 replaced."""
    rows = {
        0: [(0.4, "RICE_1KG"), (30.4, "RICE_5KG")],
        1: [(0.4, "DAL_1KG"), (60.4, "DAL_500G")],
        2: [(0.5, "RICE_1KG")],
        3: row3,
        4: [(0.5, "DAL_500G")],
        5: [(0.5, "RICE_1KG")],
    }
    return rows


def test_matching_labels_have_no_drift():
    lm = LabelMaps(SKUS, CFG.label_map)
    lm.update(robot("G1-L-04", labels_like_g1_l_04([(0.4, "RICE_1KG"), (60.4, "DAL_1KG")])))
    assert drift_rows(plan("G1-L-04"), lm.maps["G1-L-04"], CFG.label_map.drift_tol_cm) == []


def test_drift_kinds():
    lm = LabelMaps(SKUS, CFG.label_map)
    lm.update(robot("G1-L-04", labels_like_g1_l_04([(0.4, "RICE_1KG"), (85.0, "DAL_1KG")])))
    lm.update(robot("G1-L-04", {5: [(0.5, "SUGAR_1KG")]}))
    diffs = drift_rows(plan("G1-L-04"), lm.maps["G1-L-04"], CFG.label_map.drift_tol_cm)
    assert (3, 0, "RICE_1KG", "RICE_1KG", "moved") in diffs  # edge 60 -> 85 cm
    assert (5, 0, "RICE_1KG", "SUGAR_1KG", "sku") in diffs
    assert (5, 0.0, None, "SUGAR_1KG", "sku") in diffs  # sugar isn't planned on row 5


def test_drift_reported_only_after_three_days_and_once():
    lm = LabelMaps(SKUS, CFG.label_map)
    wrong = labels_like_g1_l_04([(0.4, "RICE_1KG"), (60.4, "TEA_250G")])
    out = []
    for day in range(5):
        t = T0 + timedelta(days=day)
        lm.update(robot("G1-L-04", wrong, t))
        out.append(lm.drift(plan("G1-L-04"), t))
    assert [len(x) for x in out] == [0, 0, 0, 2, 0]  # day 3: TEA where DAL is planned, + TEA extra
    first = out[3][0]
    assert isinstance(first, Drift) and first.since == T0 and first.row == 3


def test_fixed_drift_is_forgotten():
    lm = LabelMaps(SKUS, CFG.label_map)
    lm.update(robot("G1-L-04", labels_like_g1_l_04([(0.4, "RICE_1KG"), (60.4, "TEA_250G")])))
    lm.drift(plan("G1-L-04"), T0)
    lm.update(robot("G1-L-04", {3: [(0.4, "RICE_1KG"), (60.4, "DAL_1KG")]}, T0 + timedelta(1)))
    assert lm.drift(plan("G1-L-04"), T0 + timedelta(days=5)) == []


# --- inside brain.py -------------------------------------------------------------------------


def test_bay_without_planogram_is_matched_against_its_label_map(tmp_path):
    reading = robot("G6-L-02", {0: [(0.5, "DETERGENT_1KG")]})
    d = json.loads(reading.model_dump_json())
    d["rows"][0]["packs"] = [
        {
            "sku": "DETERGENT_1KG",
            "conf": 0.9,
            "x_cm": 18.0 * i,
            "w_cm": 18.0,
            "h_cm": 26.0,
            "stack": 1,
            "depth_left": 3,
        }
        for i in range(6)
    ]
    src = tmp_path / "in.jsonl"
    bus.append_jsonl(src, parse_bay_reading(d))
    s = run(src, tmp_path / "out", CFG, SKUS_PATH)
    assert not s.no_planogram and s.label_maps == {"G6-L-02"}
    (o,) = bus.read_jsonl(tmp_path / "out/observations.jsonl", SlotObservation)
    assert (o.sku, o.status, o.facings) == ("DETERGENT_1KG", "OK", 6)
    saved = parse_planogram(json.loads((tmp_path / "out/label_maps/G6-L-02.json").read_text()))
    assert saved.source == "label_map" and saved.rows[0][0].sku_id == "DETERGENT_1KG"


def test_brain_reports_persistent_drift(tmp_path):
    wrong = labels_like_g1_l_04([(0.4, "RICE_1KG"), (60.4, "TEA_250G")])
    src = tmp_path / "in.jsonl"
    bus.append_jsonl(src, [robot("G1-L-04", wrong, T0 + timedelta(days=d)) for d in range(4)])
    s = run(src, tmp_path / "out", CFG, SKUS_PATH)
    drift = bus.read_jsonl(tmp_path / "out/label_drift.jsonl", Drift)
    assert s.drift == 2 and {(x.planned, x.labelled) for x in drift} == {
        ("DAL_1KG", "TEA_250G"),
        (None, "TEA_250G"),
    }
