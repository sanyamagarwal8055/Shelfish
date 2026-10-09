"""Step 7: robot bridge on synthetic recordings and missions."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from shelfpulse.contracts import (
    UNKNOWN_SKU,
    parse_bay_reading,
    parse_mission,
    parse_robot_status,
)
from shelfpulse.layout import store_map
from shelfpulse.perception.analyze import load_pipeline
from shelfpulse.perception.identify import PlanHints
from shelfpulse.perception.privacy import blur_people
from shelfpulse.sources.robot import bridge
from shelfpulse.sources.robot.geometry import BayLocator, bay_axis
from tools.eval_readings import evaluate, load
from tools.synth.robot import record

FIXTURE = Path(__file__).parent / "fixtures" / "mission_contract_example.jsonl"
BAYS = ["G1-L-04", "G1-L-05", "G3-L-05", "G7-R-02"]
SMAP = store_map.load()


class PlannedNames:
    """Names packs from the demo planogram, so these tests need no model."""

    hints = PlanHints(["contracts/fixtures/demo_store/planograms"])

    def candidates(self, crops):
        return [None] * len(crops)

    def decide(self, cands, bay_id, row, x_cm, w_cm, ocr_words=None):
        return self.hints.planned(bay_id, row, x_cm + w_cm / 2) or UNKNOWN_SKU, 0.9


def _mission(bays, mid="M-0001", kind="mission"):
    return parse_mission(
        {"contract_version": "1.0", "mission_id": mid, "kind": kind, "bays": bays,
         "created_at": "2026-10-08T11:00:00+05:30", "speed_mps": 0.3}
    )  # fmt: skip


def _lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    rec = tmp_path_factory.mktemp("rec")
    record(rec, BAYS, seed=3, gaps=True, people={"G1-L-05"})
    out = tmp_path_factory.mktemp("out")
    pipeline = dataclasses.replace(
        load_pipeline(backend="classic", identify=False), identifier=PlannedNames()
    )
    b = bridge.run([_mission(BAYS)], out, rec, pipeline=pipeline)
    return rec, out, b


@pytest.mark.parametrize("bay_id", ["G3-L-05", "G7-R-02", "G9-E-F", "G4-E-B"])
def test_axis_matches_store_map_x_cm(bay_id):
    axis = bay_axis(SMAP.bays[bay_id])
    for x_cm in (0.0, 30.0, 120.0):
        expected = SMAP.bay_x_to_store(bay_id, x_cm)
        along = expected[1] if axis.axis == "y" else expected[0]
        assert axis.along(x_cm) == pytest.approx(along)
        assert axis.x_cm(along) == pytest.approx(x_cm)


def test_locator_sees_the_right_bay_on_the_right_side():
    loc = BayLocator(SMAP)
    axis = bay_axis(SMAP.bays["G3-L-05"])
    x, y = axis.pose_xy(60.0)
    from shelfpulse.sources.robot.geometry import Pose

    seen = loc.in_view(Pose(None, x, y, axis.heading_deg, axis.side), 60.0)
    assert [b for b, _ in seen] == ["G3-L-05"]
    other = "left" if axis.side == "right" else "right"
    assert "G3-L-05" not in [b for b, _ in loc.in_view(Pose(None, x, y, 90.0, other), 60.0)]


def test_every_bay_done_or_skipped_and_outputs_valid(recorded):
    _, out, b = recorded
    statuses = [parse_robot_status(d) for d in _lines(out / "robot_status.jsonl")]
    final = statuses[-1]
    assert final.state == "docked" and final.pending_bays == []
    assert sorted(final.done_bays + final.skipped_bays) == sorted(BAYS)
    assert final.skipped_bays == ["G1-L-05"]  # a shopper hides > 30% of it
    assert [len(s.pending_bays) for s in statuses] == [4, 3, 2, 1, 0, 0]
    assert all(a.t <= b.t for a, b in zip(statuses, statuses[1:], strict=False))
    readings = [parse_bay_reading(d) for d in _lines(out / "bay_readings.jsonl")]
    assert {r.bay_id for r in readings} == set(final.done_bays)
    assert all(r.source == "robot" for r in readings)
    for r in readings:
        assert (out / r.frame_ref).is_file()  # the (blurred) bay image was saved


def test_robot_readings_match_truth(recorded):
    rec, out, _ = recorded
    # Score the bays that were read (the skipped one has no reading by design).
    read = {r.bay_id for r in load(out / "bay_readings.jsonl")}
    gold = [g for g in load(rec / "truth.jsonl") if g.bay_id in read]
    o = evaluate(load(out / "bay_readings.jsonl"), gold, by_bay=True).overall
    assert o.facing_err == 0 and o.sku_acc == 1.0 and o.gap_recall == 1.0
    assert o.depth_n > 100 and o.depth_acc == 1.0  # robot ToF gives depth_left


def test_mission_without_recording_renders_bays(tmp_path):
    pipeline = dataclasses.replace(
        load_pipeline(backend="classic", identify=False), identifier=PlannedNames()
    )
    missions = [parse_mission(json.loads(s)) for s in FIXTURE.read_text().splitlines()]
    b = bridge.run(missions, tmp_path, None, people={"G7-R-02"}, pipeline=pipeline)
    final = b.statuses[-1]
    assert final.done_bays == ["G3-L-05", "G9-E-F"] and final.skipped_bays == ["G7-R-02"]
    assert final.mission_id == "M-0007"


def test_missing_missions_file_fails_clearly(tmp_path):
    assert bridge.main(["--missions", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path)]) == 1


def test_blur_hides_masked_pixels():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), bool)
    mask[20:60, 20:60] = True
    out = blur_people(img, mask, px_per_cm=2.0)
    assert np.abs(out[30:50, 30:50].astype(int) - img[30:50, 30:50]).mean() > 30
    assert np.array_equal(out[80:, 80:], img[80:, 80:])  # untouched far from the mask


def test_api_rejects_unknown_bays():
    from shelfpulse.sources.robot.vendor_api import FakeVendorAPI

    api = FakeVendorAPI(SMAP)
    with pytest.raises(ValueError, match="unknown bays"):
        api.send_waypoints(["G99-L-01"], 0.3, datetime.fromisoformat("2026-10-08T11:00:00+05:30"))
