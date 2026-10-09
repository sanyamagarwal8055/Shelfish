"""Step 7: SQLite store, API, and the pages it serves (run on simulated scenarios)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shelfpulse.api.server import create_app
from sim.run import simulate_with_brain
from sim.scenario import load_scenario


@cache
def client(name: str, root: Path) -> TestClient:
    out = root / name
    simulate_with_brain(load_scenario(name), out)
    return TestClient(create_app(out / "shelfpulse.db"))


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    return lambda name: client(name, root)


def test_tasks_and_verified_metrics(api):
    c = api("restock_simple")
    (t,) = c.get("/tasks").json()
    assert (t["id"], t["action"], t["status"], t["priority"]) == (
        "T-0001",
        "RESTOCK",
        "VERIFIED",
        "P1",
    )
    m = c.get("/metrics").json()
    assert m["verified_tasks"] == 1 and m["median_time_to_restore_min"] == 31.0
    assert m["open_tasks"] == 0 and m["rupees_per_h_at_risk"] == 0
    assert m["availability_pct"] == 100.0  # restocked by the end


def test_loss_prevention_note_is_never_on_staff_screens(api):
    c = api("sweep_theft")
    assert [t["action"] for t in c.get("/tasks").json()] == ["RESTOCK"]
    assert [t["action"] for t in c.get("/tasks?assignee=staff").json()] == ["RESTOCK"]
    lp = c.get("/tasks?assignee=loss_prevention").json()
    assert [t["action"] for t in lp] == ["LOSS_PREVENTION"]
    assert c.get("/metrics").json()["open_tasks"] == 1  # the note isn't counted either


def test_filters(api):
    c = api("true_stockout")
    assert [t["action"] for t in c.get("/tasks?assignee=manager&status=OPEN").json()] == ["REORDER"]
    assert c.get("/tasks?assignee=staff").json() == []


def test_feedback_round_trip_survives_brain_updates(api, tmp_path):
    c = api("misplaced")
    (t,) = c.get("/tasks").json()
    r = c.post(f"/tasks/{t['id']}/feedback", json={"verdict": "NOT_REAL", "note": "it's a tester"})
    assert r.status_code == 200 and r.json()["feedback"] == "NOT_REAL"
    (t2,) = c.get("/tasks").json()
    assert (t2["feedback"], t2["feedback_note"]) == ("NOT_REAL", "it's a tester")
    assert c.get("/metrics").json()["marked_not_real"] == 1


def test_feedback_errors(api):
    c = api("misplaced")
    assert c.post("/tasks/T-9999/feedback", json={"verdict": "DONE"}).status_code == 404
    assert c.post("/tasks/T-0001/feedback", json={"verdict": "MAYBE"}).status_code == 422


def test_store_map(api):
    m = api("restock_simple").get("/store/map").json()
    bays = {b["bay_id"]: b for b in m["bays"]}
    assert len(bays) == 220
    assert bays["G1-L-04"]["status"] == "OK"  # seen and restocked
    assert bays["G9-R-05"]["status"] == "UNSEEN"
    assert bays["G1-L-04"]["tier"] == "camera" and bays["G7-R-02"]["tier"] == "robot_only"


def test_store_map_worst_slot_wins(api):
    m = api("true_stockout").get("/store/map").json()
    assert {b["bay_id"]: b["status"] for b in m["bays"]}["G1-L-05"] == "OUT"


def test_events_and_missions(api):
    c = api("occlusion_then_robot")
    (m,) = c.get("/missions").json()
    assert m["reasons"] == {"G1-L-04": "blocked"}
    assert [e["kind"] for e in c.get("/events").json()] == ["OUT"]


def test_pages_are_served(api):
    c = api("noisy_frame")
    for path, marker in [("/", "store dashboard"), ("/staff", "Shelf tasks")]:
        r = c.get(path)
        assert (
            r.status_code == 200 and marker in r.text and "text/html" in r.headers["content-type"]
        )
