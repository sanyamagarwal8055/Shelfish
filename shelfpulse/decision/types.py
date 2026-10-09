"""Brain-internal records written under runs/<id>/ (observations, strays, events, tasks).

Not part of the Vision <-> Brain contract: only the Brain, the API and the apps read these.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from shelfpulse.contracts import BayId, SkuId, SkuRef

# UNKNOWN = slot not (fully) seen or packs not named; UNSURE = camera and robot disagree
SlotStatus = Literal["OK", "LOW", "OUT", "UNKNOWN", "UNSURE"]
StrayKind = Literal["MISPLACED", "UNKNOWN_ITEM", "AMBIGUOUS"]
EventKind = Literal["OUT", "LOW", "LOW_ESTIMATED", "MISPLACED", "UNKNOWN_ITEM", "RESOLVED"]
TaskAction = Literal["RESTOCK", "REORDER", "CYCLE_COUNT", "RETURN", "ENROL", "LOSS_PREVENTION"]
TaskStatus = Literal["OPEN", "DONE", "VERIFIED", "NOT_REAL"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlotObservation(_Model):
    """What one reading says about one planogram slot."""

    bay_id: BayId
    row: int = Field(ge=0, le=5)
    position: int = Field(ge=0)
    sku: SkuId  # the planned SKU
    status: SlotStatus
    facings: int = Field(ge=0)  # facings of the planned SKU seen (ambiguous ones included)
    planned_facings: int = Field(ge=0)
    min_facings: int = Field(ge=0)
    units: int | None = None  # sum of stack x depth_left; None if any depth is unknown
    ambiguous: bool = False  # some counted facings were AMBIGUOUS: packs
    conf: float | None = None  # lowest conf among counted facings
    source: Literal["camera", "robot"]
    t: AwareDatetime


class StrayItem(_Model):
    """A pack that is not the planned SKU of the slot it stands in."""

    kind: StrayKind
    bay_id: BayId
    row: int = Field(ge=0, le=5)
    position: int | None  # slot it stands in; None if outside every planned slot
    expected: SkuId | None  # planned SKU there
    sku: SkuRef  # what Vision saw
    x_cm: float
    conf: float
    source: Literal["camera", "robot"]
    t: AwareDatetime


class Event(_Model):
    kind: EventKind
    bay_id: BayId
    row: int = Field(ge=0, le=5)
    position: int | None = None  # planogram slot index in the row; None if off-planogram
    sku: SkuRef | None = None
    t: AwareDatetime
    since: AwareDatetime | None = None  # when the condition started
    previous: EventKind | None = None  # alert kind before this event (LOW -> OUT, OUT -> RESOLVED)
    source: Literal["camera", "robot", "estimate"]


class Task(_Model):
    """One staff (or manager) job. tasks.jsonl logs a snapshot each time a task changes."""

    id: str = Field(pattern=r"^T-\d+$")
    priority: Literal["P1", "P2", "P3"]
    action: TaskAction
    sku_id: SkuRef
    bay: BayId  # where to act (for RETURN: the bay to take the pack back to)
    row: int = Field(ge=0, le=5)  # where the problem was seen
    position: int | None = None
    seen_bay: BayId  # where the problem was seen (differs from bay for RETURN)
    qty: int = Field(ge=0)
    cause: str
    rupees_per_h: float = Field(ge=0.0)
    assignee: Literal["staff", "manager", "loss_prevention"]  # loss_prevention notes are silent
    status: TaskStatus = "OPEN"
    created_at: AwareDatetime
    updated_at: AwareDatetime
    closed_at: AwareDatetime | None = None
    time_to_restore_min: float | None = None  # alert start -> verified fix
