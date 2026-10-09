"""Step 4: camera/robot fusion, the sales-based estimate, and UNKNOWN packs giving "can't tell"."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from shelfpulse import bus
from shelfpulse.brain import load_brain_config, run
from shelfpulse.decision.store_data import StoreData
from shelfpulse.decision.types import Event, SlotObservation, Task
from shelfpulse.state.quantity import low_estimated, shelf_estimate
from shelfpulse.state.shelf_state import ShelfState
from sim.run import simulate
from sim.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[2]
SKUS = ROOT / "data" / "sku_master.csv"
IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 10, 8, 15, 0, tzinfo=IST)
CFG = load_brain_config()


def s(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# --- fusion ----------------------------------------------------------------------------------


def test_disagreement_within_window_is_a_conflict():
    st = ShelfState(conflict_window_s=300)
    assert not st.conflict("k", "camera", "OK", s(0))
    assert st.conflict("k", "robot", "OUT", s(60))  # robot says empty a minute later
    assert st.conflict("k", "camera", "OK", s(120))  # camera still says full


def test_outside_window_newest_wins():
    st = ShelfState(conflict_window_s=300)
    st.conflict("k", "robot", "OUT", s(0))
    assert not st.conflict("k", "camera", "OK", s(301))


def test_agreement_and_same_source_are_not_conflicts():
    st = ShelfState(conflict_window_s=300)
    st.conflict("k", "camera", "OUT", s(0))
    assert not st.conflict("k", "robot", "LOW", s(30))  # both say bad
    st2 = ShelfState(conflict_window_s=300)
    st2.conflict("k", "camera", "OK", s(0))
    assert not st2.conflict("k", "camera", "OUT", s(60))  # a camera changing its mind
    st3 = ShelfState(conflict_window_s=300)
    st3.conflict("k", "robot", "OUT", s(0))
    assert not st3.conflict("k", "robot", "OK", s(60))  # robot vs robot is never a conflict


def test_keys_are_independent():
    st = ShelfState(conflict_window_s=300)
    st.conflict("a", "camera", "OK", s(0))
    assert not st.conflict("b", "robot", "OUT", s(10))


# --- quantity --------------------------------------------------------------------------------


def store(on_hand: int, backroom: int) -> StoreData:
    return StoreData(snapshots={"ATTA": [(T0, on_hand, backroom)]}, data_start=T0)


def test_shelf_estimate_is_system_minus_backroom():
    assert shelf_estimate("ATTA", s(60), store(24, 10)) == 14
    assert shelf_estimate("ATTA", s(60), store(5, 10)) == 0  # never negative
    assert shelf_estimate("DAL", s(60), store(24, 10)) is None


def test_low_estimated_threshold():
    q = CFG.quantity  # lead 0.5 h, safety 2
    assert low_estimated(6, velocity=18, cfg=q)  # 6 < 18 * 0.5 + 2
    assert not low_estimated(12, velocity=18, cfg=q)
    assert not low_estimated(None, velocity=18, cfg=q)


# --- UNKNOWN packs: "can't tell", not OUT ------------------------------------------------------


def test_unnamed_packs_raise_no_out(tmp_path):
    sc = yaml.safe_load((ROOT / "sim/scenarios/noisy_frame.yaml").read_text())
    sc["events"] = [{"at": "10:02", "do": "unidentified", "sku": "RICE_1KG", "until": "10:15"}]
    path = tmp_path / "unk.yaml"
    path.write_text(yaml.safe_dump(sc))
    simulate(load_scenario(path), tmp_path / "out")
    run(tmp_path / "out/bay_readings.jsonl", tmp_path / "out", CFG, SKUS)
    events = bus.read_jsonl(tmp_path / "out/events.jsonl", Event)
    obs = bus.read_jsonl(tmp_path / "out/observations.jsonl", SlotObservation)
    tasks = bus.read_jsonl(tmp_path / "out/tasks.jsonl", Task)
    assert "OUT" not in {e.kind for e in events}
    assert not [t for t in tasks if t.action == "RESTOCK"]
    rice_while_unnamed = [o for o in obs if o.sku == "RICE_1KG" and 2 <= o.t.minute < 15]
    assert rice_while_unnamed and {o.status for o in rice_while_unnamed} == {"UNKNOWN"}
