"""Step 2b: k-of-n alerting (k=2, n=3, clear_after=2), alone and inside brain.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shelfpulse import bus
from shelfpulse.brain import load_brain_config, run
from shelfpulse.contracts import parse_bay_reading
from shelfpulse.decision.types import Event
from shelfpulse.state.slot_tracker import RESOLVED, SlotTracker

ROOT = Path(__file__).resolve().parents[2]
SKUS = ROOT / "data" / "sku_master.csv"
IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 10, 8, 11, 0, tzinfo=IST)


def feed(tracker, kinds, source="camera", key="k"):
    """One frame per minute; returns (minute, kind, previous) for each transition."""
    out = []
    for i, kind in enumerate(kinds):
        tr = tracker.update(key, kind, T0 + timedelta(minutes=i), source)
        if tr:
            out.append((i, tr.kind, tr.previous))
    return out


@pytest.fixture
def tr():
    return SlotTracker(k=2, n=3, clear_after=2)


def test_one_noisy_frame_raises_nothing(tr):
    assert feed(tr, [None, "OUT", None, None, None]) == []


def test_two_of_three_raises_once(tr):
    assert feed(tr, ["OUT", None, "OUT", "OUT", "OUT"]) == [(2, "OUT", None)]


def test_since_is_first_bad_frame(tr):
    feed(tr, [None, "OUT"])
    t = tr.update("k", "OUT", T0 + timedelta(minutes=2))
    assert t.since == T0 + timedelta(minutes=1)


def test_resolves_after_two_ok_frames(tr):
    events = feed(tr, ["OUT", "OUT", None, "OUT", None, None])
    assert events == [(1, "OUT", None), (5, RESOLVED, "OUT")]


def test_low_escalates_to_out(tr):
    assert feed(tr, ["LOW", "LOW", "OUT", "OUT"]) == [(1, "LOW", None), (3, "OUT", "LOW")]


def test_mixed_low_and_out_is_low(tr):
    assert feed(tr, ["OUT", "LOW"]) == [(1, "LOW", None)]


def test_robot_frame_decides_alone(tr):
    assert feed(tr, ["OUT"], source="robot") == [(0, "OUT", None)]
    t = tr.update("k", None, T0 + timedelta(hours=8), "robot")
    assert t.kind == RESOLVED


def test_robot_override_can_be_off():
    tr = SlotTracker(k=2, n=3, clear_after=2, robot_overrides=False)
    assert feed(tr, ["OUT"], source="robot") == []


def test_keys_are_independent(tr):
    tr.update("a", "OUT", T0)
    assert tr.update("b", "OUT", T0 + timedelta(minutes=1)) is None


def test_bad_config_and_kind():
    with pytest.raises(ValueError):
        SlotTracker(k=4, n=3, clear_after=2)
    with pytest.raises(ValueError):
        SlotTracker(k=2, n=3, clear_after=2).update("k", "EMPTY", T0)


# --- inside brain.py ------------------------------------------------------------------------


def frame(minute, rice, extra=(), occluded=()):
    """G1-L-04 row 2 (RICE_1KG across the bay, min 3) with `rice` facings."""
    packs = [
        {"sku": "RICE_1KG", "conf": 0.9, "x_cm": 10.0 * i, "w_cm": 9.5, "h_cm": 26.0,
         "stack": 1, "depth_left": 2}
        for i in range(rice)
    ] + [
        {"sku": s, "conf": 0.9, "x_cm": x, "w_cm": 6.0, "h_cm": 18.0, "stack": 1,
         "depth_left": 1}
        for s, x in extra
    ]  # fmt: skip
    return parse_bay_reading(
        {
            "bay_id": "G1-L-04",
            "source": "camera",
            "t": (T0 + timedelta(minutes=minute)).isoformat(),
            "frame_ref": "f.jpg",
            "quality": 0.9,
            "px_per_cm": 10.0,
            "rows": [{"row": 2, "occluded": [list(o) for o in occluded], "packs": packs}],
        }
    )


def events_for(tmp_path, frames):
    src = tmp_path / "in.jsonl"
    bus.append_jsonl(src, frames)
    run(src, tmp_path / "out", load_brain_config(), SKUS)
    return [
        (e.kind, e.sku, e.previous) for e in bus.read_jsonl(tmp_path / "out/events.jsonl", Event)
    ]


def test_brain_noisy_frame(tmp_path):
    assert events_for(tmp_path, [frame(0, 12), frame(1, 0), frame(2, 12)]) == []


def test_brain_out_then_restock(tmp_path):
    frames = [frame(0, 12), frame(1, 0), frame(2, 0), frame(3, 0), frame(4, 12), frame(5, 12)]
    assert events_for(tmp_path, frames) == [
        ("OUT", "RICE_1KG", None),
        ("RESOLVED", "RICE_1KG", "OUT"),
    ]


def test_brain_occluded_frames_do_not_vote(tmp_path):
    occ = [(0.0, 120.0)]
    # One visible empty frame, then two with the shelf hidden: only one vote, so no alert.
    frames = [frame(0, 12), frame(1, 0), frame(2, 0, occluded=occ), frame(3, 0, occluded=occ)]
    assert events_for(tmp_path, frames) == []


def test_brain_misplaced_raises_and_clears(tmp_path):
    shampoo = [("SHAMPOO_180ML", 115.0)]
    frames = [frame(i, 11, shampoo) for i in range(3)] + [frame(3, 12), frame(4, 12)]
    assert events_for(tmp_path, frames) == [
        ("MISPLACED", "SHAMPOO_180ML", None),
        ("RESOLVED", "SHAMPOO_180ML", "MISPLACED"),
    ]


def test_brain_ambiguous_never_alerts(tmp_path):
    amb = [("AMBIGUOUS:SOAP_100G|SHAMPOO_180ML", 115.0)]
    assert events_for(tmp_path, [frame(i, 11, amb) for i in range(4)]) == []
