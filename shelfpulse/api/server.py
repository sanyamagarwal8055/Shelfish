"""HTTP API for the dashboard and the staff page, over a run's SQLite store.

    python -m shelfpulse.api.server --db runs/<id>/shelfpulse.db [--port 8000]

then open http://localhost:8000/ (dashboard) or http://localhost:8000/staff (staff page).

    GET  /tasks?status=OPEN&assignee=staff   tasks, newest state; loss-prevention notes only
                                             when assignee=loss_prevention is asked for (silent)
    POST /tasks/{id}/feedback                {"verdict": "DONE" | "NOT_REAL", "note": "..."}
    GET  /store/map                          every bay's footprint and its worst slot status
    GET  /metrics                            availability, open tasks, Rs/h at risk, time to restore
    GET  /events, /missions                  most recent first

The pages poll every few seconds (no websocket). Feedback is stamped with the server's clock:
it records when a person pressed the button, outside the brain's reading clock.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from shelfpulse.config import REPO_ROOT
from shelfpulse.layout import store_map
from shelfpulse.storage.db import Store

APPS = REPO_ROOT / "apps"
# worst first; "UNSEEN" = no reading yet
STATUS_ORDER = ("OUT", "LOW", "UNSURE", "UNKNOWN", "OK", "UNSEEN")


class Feedback(BaseModel):
    verdict: Literal["DONE", "NOT_REAL"]
    note: str = ""


def create_app(db_path: str | Path) -> FastAPI:
    store = Store(db_path)
    smap = store_map.load()
    app = FastAPI(title="ShelfPulse")

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(APPS / "dashboard" / "index.html")

    @app.get("/staff")
    def staff() -> FileResponse:
        return FileResponse(APPS / "staff" / "index.html")

    @app.get("/tasks")
    def tasks(status: str | None = None, assignee: str | None = None) -> list[dict]:
        rows = store.tasks(status, assignee)
        if assignee != "loss_prevention":  # theft notes are silent: never on staff screens
            rows = [t for t in rows if t["assignee"] != "loss_prevention"]
        return rows

    @app.post("/tasks/{task_id}/feedback")
    def feedback(task_id: str, body: Feedback) -> dict:
        at = datetime.now().astimezone()
        if not store.feedback(task_id, body.verdict, at, body.note):
            raise HTTPException(404, f"no task {task_id}")
        return {"id": task_id, "feedback": body.verdict, "feedback_at": at.isoformat()}

    @app.get("/store/map")
    def store_map_view() -> dict:
        worst: dict[str, str] = {}
        for s in store.slots():
            cur = worst.get(s["bay_id"], "UNSEEN")
            if STATUS_ORDER.index(s["status"]) < STATUS_ORDER.index(cur):
                worst[s["bay_id"]] = s["status"]
        bays = [
            {"bay_id": b.bay_id, "run": b.run, "kind": b.kind, "tier": b.tier,
             "x0": b.x0, "x1": b.x1, "y0": b.y0, "y1": b.y1,
             "status": worst.get(b.bay_id, "UNSEEN")}
            for b in smap.bays.values()
        ]  # fmt: skip
        return {"width_m": 48.0, "depth_m": 30.0, "dock": list(smap.dock_xy), "bays": bays}

    @app.get("/metrics")
    def metrics() -> dict:
        slots = store.slots()
        seen = [s for s in slots if s["status"] in ("OK", "LOW", "OUT")]
        visible = [t for t in store.tasks() if t["assignee"] != "loss_prevention"]
        open_ = [t for t in visible if t["status"] == "OPEN"]
        verified = [t for t in visible if t["status"] == "VERIFIED"]
        restore = [
            t["time_to_restore_min"] for t in verified if t["time_to_restore_min"] is not None
        ]
        by_prio = {p: sum(t["priority"] == p for t in open_) for p in ("P1", "P2", "P3")}
        return {
            "slots_seen": len(seen),
            "availability_pct": round(100 * sum(s["status"] == "OK" for s in seen) / len(seen), 1)
            if seen else None,
            "open_tasks": len(open_),
            "open_by_priority": by_prio,
            "rupees_per_h_at_risk": round(sum(t["rupees_per_h"] for t in open_), 2),
            "verified_tasks": len(restore),
            "median_time_to_restore_min": statistics.median(restore) if restore else None,
            "marked_not_real": sum(t["feedback"] == "NOT_REAL" for t in visible),
        }  # fmt: skip

    @app.get("/events")
    def events(limit: int = 50) -> list[dict]:
        return store.events(limit)

    @app.get("/missions")
    def missions(limit: int = 20) -> list[dict]:
        return store.missions(limit)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shelfpulse.api.server", description=__doc__)
    ap.add_argument("--db", type=Path, required=True, help="runs/<id>/shelfpulse.db")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    if not args.db.exists():
        ap.error(f"{args.db} not found; run the brain or `python -m sim.run ... --brain` first")

    import uvicorn

    uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
