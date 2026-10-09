"""SQLite store for the API, dashboard and staff page (Brain-owned; *.db is git-ignored).

The brain writes every Output here as it runs: the latest verdict per slot, every event, the
latest state of each task, and every mission. Staff feedback (Done / Not real) is written back
by the API. Only metadata is stored, never images.

    slots     one row per slot: its latest SlotObservation
    events    every Event
    tasks     one row per task id: its latest Task, plus staff feedback
    missions  every Mission
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from shelfpulse.contracts import Mission, to_json_dict
from shelfpulse.decision.types import Event, SlotObservation, Task

SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
    bay_id TEXT, row INTEGER, position INTEGER, sku TEXT, status TEXT, t TEXT, doc TEXT,
    PRIMARY KEY (bay_id, row, position));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, bay_id TEXT, t TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, status TEXT, priority TEXT, assignee TEXT, bay TEXT, updated_at TEXT,
    doc TEXT, feedback TEXT, feedback_at TEXT, feedback_note TEXT);
CREATE TABLE IF NOT EXISTS missions (
    mission_id TEXT PRIMARY KEY, kind TEXT, created_at TEXT, doc TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    @classmethod
    def fresh(cls, path: str | Path) -> Store:
        """A new, empty store (an old file at `path` is replaced)."""
        Path(path).unlink(missing_ok=True)
        return cls(path)

    def close(self) -> None:
        self.db.close()

    # --- writes (brain) ----------------------------------------------------------------------

    def save(self, observations: list[SlotObservation], events: list[Event], tasks: list[Task],
             missions: list[Mission]) -> None:  # fmt: skip
        c = self.db
        c.executemany(
            "INSERT OR REPLACE INTO slots VALUES (?,?,?,?,?,?,?)",
            [(o.bay_id, o.row, o.position, o.sku, o.status, o.t.isoformat(), _doc(o))
             for o in observations],
        )  # fmt: skip
        c.executemany(
            "INSERT INTO events (kind, bay_id, t, doc) VALUES (?,?,?,?)",
            [(e.kind, e.bay_id, e.t.isoformat(), _doc(e)) for e in events],
        )
        for t in tasks:  # keep staff feedback when the brain updates a task
            c.execute(
                "INSERT INTO tasks (id, status, priority, assignee, bay, updated_at, doc) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                "priority=excluded.priority, bay=excluded.bay, updated_at=excluded.updated_at, "
                "doc=excluded.doc",
                (t.id, t.status, t.priority, t.assignee, t.bay, t.updated_at.isoformat(), _doc(t)),
            )
        c.executemany(
            "INSERT OR REPLACE INTO missions VALUES (?,?,?,?)",
            [(m.mission_id, m.kind, m.created_at.isoformat(), _doc(m)) for m in missions],
        )
        c.commit()

    # --- feedback (staff page) ---------------------------------------------------------------

    def feedback(self, task_id: str, verdict: str, at: datetime, note: str = "") -> bool:
        """Record Done / Not real for a task. False if there is no such task."""
        cur = self.db.execute(
            "UPDATE tasks SET feedback=?, feedback_at=?, feedback_note=? WHERE id=?",
            (verdict, at.isoformat(), note, task_id),
        )
        self.db.commit()
        return cur.rowcount == 1

    # --- reads (API) -------------------------------------------------------------------------

    def tasks(self, status: str | None = None, assignee: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM tasks WHERE 1=1", []
        if status:
            q, args = q + " AND status=?", [*args, status]
        if assignee:
            q, args = q + " AND assignee=?", [*args, assignee]
        rows = self.db.execute(q + " ORDER BY priority, updated_at DESC, id", args).fetchall()
        return [_task(r) for r in rows]

    def slots(self) -> list[dict]:
        return [json.loads(r["doc"]) for r in self.db.execute("SELECT doc FROM slots")]

    def events(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT doc FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [json.loads(r["doc"]) for r in rows]

    def missions(self, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            "SELECT doc FROM missions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [json.loads(r["doc"]) for r in rows]


def _doc(model) -> str:
    return json.dumps(to_json_dict(model))


def _task(r: sqlite3.Row) -> dict:
    d = json.loads(r["doc"])
    d["feedback"] = r["feedback"]
    d["feedback_at"] = r["feedback_at"]
    d["feedback_note"] = r["feedback_note"]
    return d
