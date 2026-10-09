"""Decision layer: store data, diagnosis (theft first), priority, task book and verifier."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from shelfpulse.brain import load_brain_config
from shelfpulse.decision.diagnosis import Diagnosis, diagnose
from shelfpulse.decision.priority import bucket, rupees_per_h
from shelfpulse.decision.store_data import StoreData
from shelfpulse.decision.tasks import Plan, TaskBook, plans_for_slot
from shelfpulse.decision.types import Event

IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 10, 8, 10, 0, tzinfo=IST)
CFG = load_brain_config()


def m(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def data(sales=(), snaps=()) -> StoreData:
    d = StoreData()
    for t, sku, q in sales:
        d.sales.setdefault(sku, []).append((m(t), q))
    for t, sku, on_hand, back in snaps:
        d.snapshots.setdefault(sku, []).append((m(t), on_hand, back))
    times = [m(x[0]) for x in (*sales, *snaps)]
    d.data_start = min(times) if times else None
    return d


# --- store data ------------------------------------------------------------------------------


def test_store_data_counts():
    d = data(sales=[(5, "RICE", 3), (20, "RICE", 2)], snaps=[(0, "RICE", 50, 30)])
    assert d.sold("RICE", m(0), m(10)) == 3
    assert d.system_on_hand("RICE", m(30)) == 45  # sales lower the system count
    assert d.backroom("RICE", m(30)) == 30
    assert d.system_on_hand("DAL", m(30)) is None


def test_velocity_uses_at_least_min_span():
    d = data(sales=[(5, "RICE", 6)], snaps=[(0, "RICE", 50, 30)])
    assert d.velocity("RICE", m(6), window_h=168, min_span_h=1.0) == 6.0  # not 6 / 0.1 h


def test_store_data_from_csv(tmp_path):
    (tmp_path / "pos.csv").write_text("t,sku_id,qty\n2026-10-08T10:05:00+05:30,RICE_1KG,2\n")
    (tmp_path / "inventory.csv").write_text(
        "t,sku_id,system_on_hand,backroom\n2026-10-08T10:00:00+05:30,RICE_1KG,40,10\n"
    )
    d = StoreData.from_csv(tmp_path / "pos.csv", tmp_path / "inventory.csv")
    assert d.system_on_hand("RICE_1KG", m(10)) == 38


# --- diagnosis -------------------------------------------------------------------------------

HISTORY = [(m(i), 8) for i in range(5)] + [(m(5), 4), (m(6), 0)]  # 8 packs, gone in 2 minutes


def diag(history, d, shelf_units=0):
    return diagnose("RICE", m(6), m(7), history, shelf_units, d, CFG.diagnosis)


def test_theft_wins_over_full_backroom():
    d = data(snaps=[(0, "RICE", 100, 50)])
    r = diag(HISTORY, d)
    assert (r.cause, r.theft_units, r.backroom) == ("THEFT", 8, 50)


def test_a_drop_that_was_sold_is_not_theft():
    d = data(sales=[(5, "RICE", 4), (6, "RICE", 4)], snaps=[(0, "RICE", 100, 50)])
    assert diag(HISTORY, d).cause == "REPLENISHMENT_GAP"


def test_phantom_vs_true_stockout():
    empty = [(m(i), 0) for i in range(7)]
    assert diag(empty, data(snaps=[(0, "RICE", 18, 0)])).cause == "PHANTOM_STOCK"
    assert diag(empty, data(snaps=[(0, "RICE", 0, 0)])).cause == "TRUE_STOCKOUT"
    # 18 on the books but 16 of them are on other shelves: a real stockout, not a wrong count
    assert diag(empty, data(snaps=[(0, "RICE", 18, 0)]), shelf_units=16).cause == "TRUE_STOCKOUT"


def test_no_store_data():
    assert diag(HISTORY, None).cause == "NO_STORE_DATA"


# --- priority --------------------------------------------------------------------------------


def test_rupees_and_buckets():
    p = CFG.priority
    assert rupees_per_h("OUT", 84, 14, p) == 1176.0
    assert rupees_per_h("LOW", 84, 14, p) == 588.0
    assert bucket(1176, "RESTOCK", p) == "P1"
    assert bucket(50, "RESTOCK", p) == "P2"
    assert bucket(0, "RESTOCK", p) == "P3"
    assert bucket(0, "LOSS_PREVENTION", p) == "P1"  # floor
    assert bucket(0, "CYCLE_COUNT", p) == "P2"


def test_fast_seller_outranks_slow_mover():
    p = CFG.priority
    fast, slow = rupees_per_h("OUT", 30, 14, p), rupees_per_h("OUT", 1, 14, p)
    assert bucket(fast, "RESTOCK", p) < bucket(slow, "RESTOCK", p)  # "P1" < "P3"


# --- task book -------------------------------------------------------------------------------


def ev(kind, minute, sku="RICE", position=0, previous=None, since=None):
    return Event(kind=kind, bay_id="G1-L-04", row=2, position=position, sku="RICE_1KG"
                 if sku == "RICE" else sku, t=m(minute), since=since or m(minute),
                 previous=previous, source="camera")  # fmt: skip


def prio(action, rupees):
    return bucket(rupees, action, CFG.priority)


def test_theft_plans_note_first_then_restock():
    plans = plans_for_slot(Diagnosis("THEFT", theft_units=8, backroom=50), "G1-L-04", 21, 0, prio)
    assert [p.action for p in plans] == ["LOSS_PREVENTION", "RESTOCK"]
    assert plans[0].assignee == "loss_prevention"


def test_true_stockout_plans_reorder_not_restock():
    plans = plans_for_slot(Diagnosis("TRUE_STOCKOUT", backroom=0), "G1-L-05", 12, 0, prio)
    assert [p.action for p in plans] == ["REORDER"]


def restock(qty=84):
    return [Plan("RESTOCK", "G1-L-04", qty, "REPLENISHMENT_GAP", 1176, "P1", "staff")]


def test_book_dedups_and_verifies():
    book = TaskBook()
    (t1,) = book.upsert(ev("LOW", 1), restock(40))
    (t2,) = book.upsert(ev("OUT", 3, previous="LOW", since=m(1)), restock(84))
    assert t1.id == t2.id and t2.qty == 84  # escalation updates the same task
    (done,) = book.verify(ev("RESOLVED", 31, previous="OUT", since=m(1)))
    assert (done.status, done.time_to_restore_min) == ("VERIFIED", 30.0)
    assert book.open == {}


def test_reorder_is_shared_by_slots_of_one_sku():
    book = TaskBook()
    plan = [Plan("REORDER", "G1-L-05", 12, "TRUE_STOCKOUT", 0, "P2", "manager")]
    (a,) = book.upsert(ev("OUT", 1, sku="ATTA_5KG", position=0), plan)
    (b,) = book.upsert(ev("OUT", 1, sku="ATTA_5KG", position=1), plan)
    assert a.id == b.id


def test_loss_prevention_note_is_not_auto_closed():
    book = TaskBook()
    note = [Plan("LOSS_PREVENTION", "G1-L-04", 8, "THEFT", 0, "P1", "loss_prevention")]
    book.upsert(ev("OUT", 1), note + restock())
    closed = book.verify(ev("RESOLVED", 20, previous="OUT", since=m(1)))
    assert [t.action for t in closed] == ["RESTOCK"]
    assert [t.action for t in book.open.values()] == ["LOSS_PREVENTION"]


def test_resolving_a_stray_does_not_close_slot_tasks():
    book = TaskBook()
    book.upsert(ev("OUT", 1), restock())
    stray = ev("RESOLVED", 5, sku="SHAMPOO_180ML", previous="MISPLACED")
    assert book.verify(stray) == []


@pytest.mark.parametrize("no_data", [True, False])
def test_brain_without_store_data_still_tasks(tmp_path, no_data):
    from shelfpulse import bus
    from shelfpulse.brain import run
    from shelfpulse.config import REPO_ROOT
    from shelfpulse.decision.types import Task
    from sim.run import simulate
    from sim.scenario import load_scenario

    simulate(load_scenario("restock_simple"), tmp_path)
    skus = REPO_ROOT / "data" / "sku_master.csv"
    s = run(tmp_path / "bay_readings.jsonl", tmp_path, CFG, skus, store_data_dir=not no_data)
    tasks = bus.read_jsonl(tmp_path / "tasks.jsonl", Task)
    assert tasks[0].action == "RESTOCK"
    assert tasks[0].cause == ("NO_STORE_DATA" if no_data else "REPLENISHMENT_GAP")
    assert s.store_data is (not no_data)
