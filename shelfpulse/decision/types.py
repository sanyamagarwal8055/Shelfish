"""Brain-internal records written to runs/<id>/events.jsonl and tasks.jsonl.

Not part of the Vision <-> Brain contract: only the Brain, the API and the apps read these.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from shelfpulse.contracts import BayId, SkuRef

EventKind = Literal["OUT", "LOW", "LOW_ESTIMATED", "MISPLACED", "UNKNOWN_ITEM", "RESOLVED"]
TaskAction = Literal["RESTOCK", "REORDER", "CYCLE_COUNT", "RETURN", "ENROL", "LOSS_PREVENTION"]
TaskStatus = Literal["OPEN", "DONE", "VERIFIED", "NOT_REAL"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Event(_Model):
    kind: EventKind
    bay_id: BayId
    row: int = Field(ge=0, le=5)
    position: int | None = None  # planogram slot index in the row; None if off-planogram
    sku: SkuRef | None = None
    t: AwareDatetime
    since: AwareDatetime | None = None  # when the condition started
    source: Literal["camera", "robot", "estimate"]


class Task(_Model):
    id: str = Field(pattern=r"^T-\d+$")
    priority: Literal["P1", "P2", "P3"]
    action: TaskAction
    sku_id: SkuRef
    bay: BayId
    qty: int = Field(ge=0)
    cause: str
    rupees_per_h: float = Field(ge=0.0)
    status: TaskStatus = "OPEN"
    created_at: AwareDatetime
    closed_at: AwareDatetime | None = None
