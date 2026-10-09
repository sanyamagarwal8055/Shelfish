"""Turn diagnosed alerts into tasks, keep one open task per problem, verify fixes.

Actions by cause:

    THEFT              LOSS_PREVENTION note (silent, to loss prevention), then the restock
                       logic below, so the shelf still gets refilled
    REPLENISHMENT_GAP  RESTOCK from the backroom (staff)
    PHANTOM_STOCK      CYCLE_COUNT: the system count is wrong (manager)
    TRUE_STOCKOUT      REORDER (manager); no restock, there is nothing to restock
    NO_STORE_DATA      RESTOCK (staff): the shelf is empty, cause unknown
    MISPLACED          RETURN the pack to its home bay (staff)
    UNKNOWN_ITEM       ENROL the product: photos for the gallery (staff)

A problem is (bay, row, position, sku, family), family "slot" for OUT/LOW and "stray" for a
misplaced/unknown pack. A repeat alert for an open problem (LOW -> OUT) updates its task.
REORDER and CYCLE_COUNT concern the whole SKU, so two empty slots of one SKU share one task.
When the tracker resolves a problem, its open tasks become VERIFIED with time-to-restore,
except loss-prevention notes, which people close.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from shelfpulse.decision.diagnosis import Diagnosis
from shelfpulse.decision.types import Event, Task

SLOT_KINDS = ("OUT", "LOW", "LOW_ESTIMATED")
STRAY_KINDS = ("MISPLACED", "UNKNOWN_ITEM")
CLOSED_BY_VERIFIER = ("RESTOCK", "REORDER", "CYCLE_COUNT", "RETURN", "ENROL")
SKU_LEVEL = ("REORDER", "CYCLE_COUNT")

ProblemKey = tuple[str, int, int | None, str, str]


def problem_key(e: Event) -> ProblemKey:
    kind = e.previous if e.kind == "RESOLVED" else e.kind
    family = "slot" if kind in SLOT_KINDS else "stray"
    return (e.bay_id, e.row, e.position, e.sku, family)


@dataclass(frozen=True)
class Plan:
    """What a task should say; the book turns it into a Task with an id and times."""

    action: str
    bay: str
    qty: int
    cause: str
    rupees_per_h: float
    priority: str
    assignee: str


def _index(e: Event, action: str) -> tuple:
    return ((e.sku,), action) if action in SKU_LEVEL else (problem_key(e), action)


@dataclass
class TaskBook:
    _next: int = 1
    open: dict[tuple, Task] = field(default_factory=dict)

    def upsert(self, e: Event, plans: list[Plan]) -> list[Task]:
        """Create or update the tasks for an alert. Returns the snapshots to log."""
        out = []
        for p in plans:
            current = self.open.get(_index(e, p.action))
            fields = dict(priority=p.priority, bay=p.bay, qty=p.qty, cause=p.cause,
                          rupees_per_h=p.rupees_per_h, updated_at=e.t)  # fmt: skip
            if current is None:
                task = Task(
                    id=f"T-{self._next:04d}",
                    action=p.action,
                    sku_id=e.sku,
                    row=e.row,
                    position=e.position,
                    seen_bay=e.bay_id,
                    assignee=p.assignee,
                    created_at=e.t,
                    **fields,
                )
                self._next += 1
            else:
                task = current.model_copy(update=fields)
            self.open[_index(e, p.action)] = task
            out.append(task)
        return out

    def verify(self, e: Event) -> list[Task]:
        """Close the problem's open tasks when the tracker says it is fixed."""
        out = []
        for idx, task in list(self.open.items()):
            if task.action in CLOSED_BY_VERIFIER and idx == _index(e, task.action):
                out.append(_close(task, e.t, e.since))
                del self.open[idx]
        return out


def _close(task: Task, t: datetime, since: datetime | None) -> Task:
    restore = (t - since).total_seconds() / 60 if since else None
    return task.model_copy(
        update=dict(status="VERIFIED", closed_at=t, updated_at=t, time_to_restore_min=restore)
    )


def plans_for_slot(
    d: Diagnosis, bay: str, need: int, rupees: float, prio: Callable[[str, float], str]
) -> list[Plan]:
    """Plans for an OUT/LOW alert. `need`: packs to fill the slot; `prio(action, rupees)`."""

    def plan(action: str, qty: int, assignee: str) -> Plan:
        return Plan(action, bay, qty, d.cause, rupees, prio(action, rupees), assignee)

    plans = []
    if d.cause == "THEFT":
        plans.append(plan("LOSS_PREVENTION", d.theft_units, "loss_prevention"))
        if d.backroom:  # still refill the shelf; unknown backroom: just the note
            plans.append(plan("RESTOCK", min(d.backroom, need), "staff"))
        elif d.backroom == 0:
            plans.append(plan("REORDER", need, "manager"))
    elif d.cause == "REPLENISHMENT_GAP":
        plans.append(plan("RESTOCK", min(d.backroom or 0, need), "staff"))
    elif d.cause == "PHANTOM_STOCK":
        plans.append(plan("CYCLE_COUNT", d.phantom_units, "manager"))
    elif d.cause == "TRUE_STOCKOUT":
        plans.append(plan("REORDER", need, "manager"))
    else:  # NO_STORE_DATA
        plans.append(plan("RESTOCK", need, "staff"))
    return plans
