"""Step 3: the store simulator, and what the brain makes of each scenario so far.

Each scenario's `expect:` says the full target outcome. These tests check the parts that exist
today (readings, POS, stock, events); tasks, fusion, quantity and missions add their checks as
those steps land.
"""

from __future__ import annotations

import csv
from datetime import timedelta
from functools import cache
from pathlib import Path

import pytest

from shelfpulse import bus
from shelfpulse.brain import load_brain_config, run
from shelfpulse.contracts import BayReading, check_skus, reading_skus
from shelfpulse.decision.types import Event, StrayItem, Task
from sim.run import simulate
from sim.scenario import load_scenario, scenario_names
from sim.store import spread

ROOT = Path(__file__).resolve().parents[2]
SKUS = ROOT / "data" / "sku_master.csv"


class Result:
    def __init__(self, name: str, out: Path):
        self.sc = load_scenario(name)
        self.store = simulate(self.sc, out)
        run(out / "bay_readings.jsonl", out, load_brain_config(), SKUS)
        self.readings = bus.read_jsonl(out / "bay_readings.jsonl", BayReading)
        self.events = bus.read_jsonl(out / "events.jsonl", Event)
        self.strays = bus.read_jsonl(out / "strays.jsonl", StrayItem)
        log = bus.read_jsonl(out / "tasks.jsonl", Task)
        self.task_log = log
        self.tasks = list({t.id: t for t in log}.values())  # final state of each task
        with open(out / "pos.csv", encoding="utf-8") as f:
            self.pos = list(csv.DictReader(f))
        with open(out / "inventory.csv", encoding="utf-8") as f:
            self.inventory = list(csv.DictReader(f))

    def kinds(self, **match) -> list[str]:
        return [e.kind for e in self.events if all(getattr(e, k) == v for k, v in match.items())]

    def minute(self, hhmm: str):
        return self.sc.at(hhmm)


@cache
def result(name: str, root: Path) -> Result:
    return Result(name, root / name)


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    return lambda name: result(name, root)


# --- every scenario ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", scenario_names())
def test_scenario_runs_and_obeys_contract(name, sim):
    r = sim(name)
    assert r.readings, "no readings"
    assert r.sc.expect, "say what the scenario should produce"
    for rd in r.readings:  # parsed by bus.read_jsonl, so valid; also no unknown SKUs
        assert not check_skus(reading_skus(rd), set(r.store.skus))


def test_eleven_scenarios():
    assert len(scenario_names()) == 11


def test_deterministic(tmp_path):
    sc = load_scenario("facing_up")
    a = simulate(sc, tmp_path / "a")
    b = simulate(sc, tmp_path / "b")
    assert (tmp_path / "a/bay_readings.jsonl").read_bytes() == (
        tmp_path / "b/bay_readings.jsonl"
    ).read_bytes()
    assert a.pos == b.pos


def test_one_camera_frame_per_minute(sim):
    r = sim("restock_simple")
    cams = [x for x in r.readings if x.source == "camera"]
    assert len(cams) == 60
    assert all(b.t - a.t == timedelta(minutes=1) for a, b in zip(cams, cams[1:], strict=False))


def test_spread_round_robin():
    assert spread(5, 3, 4) == [2, 2, 1]
    assert spread(20, 3, 4) == [4, 4, 4]  # capped at capacity


# --- scenario outcomes so far -----------------------------------------------------------------

RICE_ROW2 = {"bay_id": "G1-L-04", "row": 2, "position": 0}


def test_restock_simple(sim):
    r = sim("restock_simple")
    out, resolved = [e for e in r.events if e.row == 2]
    assert (out.kind, out.sku) == ("OUT", "RICE_1KG")
    assert out.since == r.minute("10:05") and out.t == r.minute("10:06")
    assert resolved.kind == "RESOLVED" and resolved.t == r.minute("10:36")
    assert resolved.t - out.since == timedelta(minutes=31)  # time to restore
    assert sum(int(p["qty"]) for p in r.pos) == 84
    assert r.store.backroom["RICE_1KG"] == 100 - 84


def test_true_stockout(sim):
    r = sim("true_stockout")
    assert r.kinds(sku="ATTA_5KG") == ["OUT", "OUT"]
    assert r.store.system["ATTA_5KG"] == 0 and r.store.backroom["ATTA_5KG"] == 0


def test_phantom_stock(sim):
    r = sim("phantom_stock")
    assert r.kinds(sku="ATTA_5KG") == ["OUT", "OUT"]
    assert r.pos == []
    last = [row for row in r.inventory if row["sku_id"] == "ATTA_5KG"][-1]
    assert (last["system_on_hand"], last["backroom"]) == ("18", "0")


def test_sweep_theft(sim):
    r = sim("sweep_theft")
    assert r.kinds(row=0, sku="RICE_1KG") == ["OUT"]
    assert r.pos == []  # packs vanished with no sale


def test_misplaced(sim):
    r = sim("misplaced")
    assert [(e.kind, e.sku) for e in r.events] == [
        ("MISPLACED", "SHAMPOO_180ML"),
        ("RESOLVED", "SHAMPOO_180ML"),
    ]


def test_noisy_frame(sim):
    r = sim("noisy_frame")
    assert r.events == []
    empty = [x for x in r.readings if x.t == r.minute("10:05")]
    assert empty and not empty[0].rows[2].packs  # the bad frame really was empty


def test_occlusion_no_alert_while_hidden(sim):
    r = sim("occlusion_then_robot")
    assert all(e.t >= r.minute("10:40") for e in r.events)
    hidden = [x for x in r.readings if r.minute("10:05") <= x.t < r.minute("10:40")]
    assert hidden and all(row.occluded == [[0.0, 120.0]] for x in hidden for row in x.rows)


def test_peak_hours_no_alert_while_hidden(sim):
    assert sim("peak_hours").events == []


def test_ambiguous_never_misplaced(sim):
    r = sim("ambiguous")
    assert "MISPLACED" not in r.kinds()
    assert r.events == []
    assert {s.kind for s in r.strays} == {"AMBIGUOUS"}


def test_facing_up_front_looks_full(sim):
    r = sim("facing_up")
    last = r.readings[-1].rows[2]
    assert len(last.packs) == 12 and all(p.depth_left is None for p in last.packs)
    assert sum(int(p["qty"]) for p in r.pos) == 28
    assert r.store.slots[("G1-L-04", 2, 0)].units == 12


def test_conflict_robot_sees_empty(sim):
    r = sim("conflict")
    robot = [x for x in r.readings if x.source == "robot"]
    assert len(robot) == 1 and robot[0].t == r.minute("15:00")
    assert robot[0].rows[2].packs == [] and robot[0].rows[2].labels
    cam = [x for x in r.readings if x.source == "camera" and x.t == r.minute("15:00")]
    assert len(cam[0].rows[2].packs) == 12


@pytest.mark.parametrize(
    "bad",
    [
        {"do": "sell", "bay": "G1-L-04", "row": 2, "position": 9, "qty": 1},
        {"do": "occlude", "bay": "G1-L-04", "x": [50.0, 40.0]},
        {"do": "sell", "bay": "G1-L-04"},
    ],
)
def test_bad_events_rejected(tmp_path, bad):
    import yaml

    sc = yaml.safe_load((ROOT / "sim/scenarios/noisy_frame.yaml").read_text())
    sc["events"] = [{"at": "10:05", **bad}]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(sc))
    with pytest.raises(ValueError):
        simulate(load_scenario(path), tmp_path / "out")


# --- tasks (decision layer) ----------------------------------------------------------------


def summary(r: Result) -> list[tuple]:
    return [(t.action, t.sku_id, t.bay, t.status, t.priority) for t in r.tasks]


def test_restock_simple_task(sim):
    r = sim("restock_simple")
    (t,) = r.tasks
    assert (t.action, t.priority, t.cause, t.status) == ("RESTOCK", "P1", "REPLENISHMENT_GAP",
                                                         "VERIFIED")  # fmt: skip
    assert t.qty == 84 and t.time_to_restore_min == 31.0


def test_true_stockout_task(sim):
    assert summary(sim("true_stockout")) == [("REORDER", "ATTA_5KG", "G1-L-05", "OPEN", "P1")]


def test_phantom_stock_task(sim):
    (t,) = sim("phantom_stock").tasks
    assert (t.action, t.qty, t.assignee) == ("CYCLE_COUNT", 18, "manager")


def test_sweep_theft_tasks(sim):
    r = sim("sweep_theft")
    assert [(t.action, t.assignee) for t in r.tasks] == [
        ("LOSS_PREVENTION", "loss_prevention"),  # decided before the restock
        ("RESTOCK", "staff"),
    ]
    assert r.tasks[0].qty == 8 and r.tasks[0].priority == "P1"


def test_misplaced_task(sim):
    (t,) = sim("misplaced").tasks
    assert (t.action, t.sku_id, t.bay, t.seen_bay, t.status) == (
        "RETURN", "SHAMPOO_180ML", "G7-R-02", "G1-L-04", "VERIFIED",
    )  # fmt: skip


@pytest.mark.parametrize("name", ["noisy_frame", "ambiguous", "peak_hours"])
def test_no_tasks(sim, name):
    assert sim(name).tasks == []


def test_unidentified_packs_read_as_unknown(tmp_path):
    import yaml

    sc = yaml.safe_load((ROOT / "sim/scenarios/noisy_frame.yaml").read_text())
    sc["events"] = [{"at": "10:05", "do": "unidentified", "sku": "RICE_1KG", "until": "10:10"}]
    path = tmp_path / "unk.yaml"
    path.write_text(yaml.safe_dump(sc))
    simulate(load_scenario(path), tmp_path / "out")
    rows = [
        r.rows[2]
        for r in bus.read_jsonl(tmp_path / "out/bay_readings.jsonl", BayReading)
        if r.t.minute in (5, 9)
    ]
    assert rows and all({p.sku for p in row.packs} == {"UNKNOWN"} for row in rows)
