"""Step 5: robot planner (queue, route, scheduler gating) and the committed missions fixture."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shelfpulse import bus
from shelfpulse.brain import load_brain_config
from shelfpulse.config import load_robot_config
from shelfpulse.contracts import Mission, RobotStatus
from shelfpulse.decision.store_data import StoreData
from shelfpulse.layout import store_map
from shelfpulse.robot_planner import route
from shelfpulse.robot_planner.mission_queue import MissionQueue
from shelfpulse.robot_planner.scheduler import Scheduler

ROOT = Path(__file__).resolve().parents[2]
MISSIONS = ROOT / "contracts" / "fixtures" / "missions"
IST = timezone(timedelta(hours=5, minutes=30))
CFG = load_brain_config()
ROBOT = load_robot_config()
SMAP = store_map.load()


def at(hh: int, mm: int = 0) -> datetime:
    return datetime(2026, 10, 8, hh, mm, tzinfo=IST)


def scheduler(store_data: StoreData | None = None) -> Scheduler:
    q = MissionQueue(CFG.planner)
    return Scheduler(ROBOT, CFG.planner, CFG.missions.footfall_tx_per_10min_max, SMAP, q,
                     store_data)  # fmt: skip


# --- queue -----------------------------------------------------------------------------------


def test_blocked_is_due_at_once_others_wait():
    q = MissionQueue(CFG.planner)
    q.add("G3-L-05", "ambiguous", at(10))
    assert not q.ready(at(10, 4))
    assert q.ready(at(10, 5))
    q2 = MissionQueue(CFG.planner)
    q2.add("G1-L-04", "blocked", at(10))
    assert q2.ready(at(10))


def test_pop_most_urgent_first_and_keeps_higher_reason():
    q = MissionQueue(CFG.planner)
    q.add("G3-L-05", "ambiguous", at(10))
    q.add("G1-L-04", "verify", at(10))
    q.add("G3-L-05", "blocked", at(10, 1))  # upgrades the bay's reason
    assert [(e.bay_id, e.reason) for e in q.pop(8, at(10, 10))] == [
        ("G3-L-05", "blocked"),
        ("G1-L-04", "verify"),
    ]
    assert q.entries == {}


def test_cooldown_after_a_visit_except_conflict():
    q = MissionQueue(CFG.planner)
    q.visited("G1-L-04", at(10))
    assert not q.add("G1-L-04", "blocked", at(10, 30))
    assert not q.add("G1-L-04", "ambiguous", at(10, 30))
    assert q.add("G1-L-04", "conflict", at(10, 30))  # the robot's own look caused it
    q.drop("G1-L-04")
    assert q.add("G1-L-04", "blocked", at(11, 1))  # cooldown over


def test_not_before():
    q = MissionQueue(CFG.planner)
    q.add("G7-R-02", "verify", at(10), not_before=at(10, 30))
    assert not q.ready(at(10, 29))
    assert q.ready(at(10, 35))  # waited 5 min after not_before


# --- route -----------------------------------------------------------------------------------


def test_route_visits_every_bay_and_is_shortest():
    from itertools import permutations

    bays = ["G9-R-05", "G1-L-04", "G6-R-06", "G3-L-05", "G9-E-F"]
    ordered = route.order(bays, SMAP)
    assert sorted(ordered) == sorted(bays)
    best = min(route.walk_m(list(p), SMAP) for p in permutations(bays))
    assert route.walk_m(ordered, SMAP) == pytest.approx(best)  # 2-opt finds the optimum here


def test_route_trivial():
    assert route.order([], SMAP) == []
    assert route.order(["G1-L-04"], SMAP) == ["G1-L-04"]


# --- scheduler gating ------------------------------------------------------------------------


def test_sweeps_at_seven_and_three_only_once():
    s = scheduler()
    (m,) = s.tick(at(7))
    assert (m.kind, m.bays, m.speed_mps) == ("sweep", [], ROBOT.speed_sweep_mps)
    assert s.tick(at(7, 1)) == []
    s.busy_until = None
    assert s.tick(at(7, 2)) == []  # already swept today
    assert s.tick(at(14, 59)) == []
    assert [m.kind for m in s.tick(at(15))] == ["sweep"]


def test_late_start_skips_a_missed_sweep():
    assert scheduler().tick(at(10)) == []  # 07:00 was long ago


def test_mission_respects_max_bays_and_route():
    s = scheduler()
    bays = [f"G{r}-L-0{i}" for r in (6, 7) for i in range(1, 6)]  # 10 robot-only bays
    for b in bays:
        s.queue.add(b, "blocked", at(10))
    (m,) = s.tick(at(10))
    assert m.kind == "mission" and len(m.bays) == ROBOT.missions.max_bays
    assert m.speed_mps == ROBOT.speed_mission_mps
    assert set(m.reasons) == set(m.bays) and len(s.queue.entries) == 2


def test_no_mission_while_robot_busy_then_after():
    s = scheduler()
    s.queue.add("G1-L-04", "blocked", at(10))
    (m,) = s.tick(at(10))
    s.queue.add("G3-L-05", "blocked", at(10, 1))
    assert s.tick(at(10, 1)) == []  # still out on M-0001
    s.on_status(RobotStatus(t=at(10, 3), state="docked", x_m=42.2, y_m=4.8,
                            mission_id=m.mission_id, done_bays=["G1-L-04"]))  # fmt: skip
    assert [x.bays for x in s.tick(at(10, 4))] == [["G3-L-05"]]


def test_blackout_holds_missions_until_21():
    s = scheduler()
    s.queue.add("G1-L-04", "blocked", at(18, 45))
    for hh, mm in [(18, 45), (19, 0), (20, 59)]:
        assert s.tick(at(hh, mm)) == []
    assert len(s.tick(at(21))) == 1


def test_max_two_missions_per_hour():
    s = scheduler()
    sent = []
    for minute in range(0, 60, 5):
        s.queue.add(f"G7-R-0{minute // 5 % 9 + 1}", "blocked", at(10, minute))
        s.busy_until = None  # robot instantly back
        sent += s.tick(at(10, minute))
    assert len(sent) == ROBOT.missions.max_per_hour


def test_busy_store_holds_missions():
    sales = [(at(10, 0) + timedelta(seconds=10 * i), 1) for i in range(30)]  # 30 sales in 5 min
    s = scheduler(StoreData(sales={"RICE_1KG": sales}, data_start=at(10)))
    s.queue.add("G1-L-04", "blocked", at(10, 5))
    assert s.tick(at(10, 5)) == []  # > 25 transactions in the last 10 min
    assert len(s.tick(at(10, 16))) == 1  # the rush is over


def test_skipped_bays_are_requeued():
    s = scheduler()
    s.on_status(RobotStatus(t=at(10), state="docked", x_m=42.2, y_m=4.8, mission_id="M-0001",
                            skipped_bays=["G1-L-05"]))  # fmt: skip
    assert s.queue.entries["G1-L-05"].reason == "blocked"


# --- fixture for the Vision robot bridge -----------------------------------------------------


@pytest.mark.parametrize("path", sorted(MISSIONS.glob("*.jsonl")), ids=lambda p: p.name)
def test_missions_fixture(path):
    ms = bus.read_jsonl(path, Mission)
    assert ms and {m.kind for m in ms} == {"sweep", "mission"}
    for m in ms:
        if m.kind == "mission":
            assert 1 <= len(m.bays) <= ROBOT.missions.max_bays
            assert m.bays == route.order(m.bays, SMAP)  # planner order
