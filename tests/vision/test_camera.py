"""Step 8: camera service on a short synthetic time-lapse."""

from __future__ import annotations

import pytest

from shelfpulse.contracts import parse_bay_reading
from shelfpulse.perception.analyze import load_pipeline
from shelfpulse.sources.camera import service
from tools.eval_readings import evaluate, load
from tools.synth.sequence import Story, left_to_right, make_sequence

# RICE_1KG slot on G1-L-04 row 0 empties within ~3 minutes, restocked at minute 6; a shopper
# stands in front of G1-L-05 during minutes 2-4.
STORY = Story(
    sell_out=("G1-L-04", 0, 0, 0),
    restock=("G1-L-04", 6),
    occlude=("G1-L-05", 2, 5, 30.0, 90.0),
    background_per_hour=0.0,
    sell_per_minute=8,
)
CAMERAS = ["G1-L-C2", "G1-L-C3"]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    seq = tmp_path_factory.mktemp("seq")
    make_sequence(seq, 8, seed=2, cameras=CAMERAS, story=STORY)
    out = tmp_path_factory.mktemp("cam")
    svc = service.run_folder(
        seq / "frames", out, pipeline=load_pipeline(backend="classic", identify=False)
    )
    return seq, out, svc


def _by_minute(path):
    return {(r.bay_id, r.t.minute): r for r in load(path)}


def _slot_empty(r) -> bool:
    row = next((x for x in r.rows if x.row == 0), None)
    return row is not None and any(
        g.x_cm < 30 and g.x_cm + g.w_cm > 0 and g.w_cm > 10 for g in row.gaps
    )


def test_camera_sees_bays_left_to_right():
    # Face L is seen looking east: the bay further back (higher number) is on the left.
    assert left_to_right(["G1-L-03", "G1-L-04"]) == ["G1-L-04", "G1-L-03"]
    assert left_to_right(["G1-R-04", "G1-R-03"]) == ["G1-R-03", "G1-R-04"]


def test_one_valid_camera_reading_per_bay_per_minute(ran):
    seq, out, svc = ran
    lines = [parse_bay_reading(r.model_dump(mode="json")) for r in load(out / "bay_readings.jsonl")]
    assert len(lines) == 8 * 4  # 8 minutes x 2 cameras x 2 bays
    assert all(r.source == "camera" and all(row.labels == [] for row in r.rows) for r in lines)
    assert all((out / r.frame_ref).is_file() for r in lines)


def test_gap_appears_and_clears_at_the_right_minutes(ran):
    seq, out, _ = ran
    g, p = _by_minute(seq / "truth.jsonl"), _by_minute(out / "bay_readings.jsonl")
    truth = [_slot_empty(g[("G1-L-04", m)]) for m in range(8)]
    pred = [_slot_empty(p[("G1-L-04", m)]) for m in range(8)]
    assert any(truth) and not truth[0] and not truth[-1]  # empties, then restocked
    assert pred == truth


def test_shopper_minutes_are_occluded_and_lower_quality(ran):
    seq, out, _ = ran
    p = _by_minute(out / "bay_readings.jsonl")
    occ = [any(r.occluded for r in p[("G1-L-05", m)].rows) for m in range(8)]
    assert occ == [m in (2, 3, 4) for m in range(8)]
    assert all(p[("G1-L-05", m)].quality < 0.9 for m in (2, 3, 4))
    assert p[("G1-L-05", 0)].quality > 0.9


def test_counts_match_truth(ran):
    seq, out, _ = ran
    o = evaluate(load(out / "bay_readings.jsonl"), load(seq / "truth.jsonl"), by_time=True).overall
    assert o.facing_err <= 0.3 and o.gap_recall >= 0.9


def test_missing_calibration_fails_clearly(tmp_path):
    (tmp_path / "frames").mkdir()
    assert service.main(["--input", str(tmp_path / "frames"), "--out", str(tmp_path / "o")]) == 1
