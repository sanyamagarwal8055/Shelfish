"""Bays that need a second look from the robot, and how urgently.

One entry per bay; a new reason for a queued bay keeps the higher priority. An entry is ready
once it has waited `ready_after_min`, or at once if its priority reaches `ready_priority`
(a blocked camera bay can't wait). After the robot reads a bay, its entry is dropped, and for
`requeue_cooldown_min` the bay can't be queued again for a `cooldown_reasons` reason (a camera
still blocked, packs still ambiguous: the robot's fresh look already answers those), so one
stubborn bay can't eat every mission. Other reasons, such as a conflict the robot's own reading
just caused, queue at once. An entry may carry `not_before` (e.g. verify a restock later).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field


class PlannerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reason_priority: dict[str, int]
    ready_after_min: float = Field(ge=0)
    ready_priority: int
    requeue_cooldown_min: float = Field(ge=0)
    cooldown_reasons: list[str]
    verify_after_min: float = Field(ge=0)  # verify an off-camera restock this long after the task
    sweep_grace_min: float = Field(gt=0)  # a sweep still starts if the clock is this late
    sweep_duration_min: float = Field(gt=0)  # robot is busy this long after a sweep starts
    bay_scan_s: float = Field(ge=0)  # time to film one bay on a mission


@dataclass
class Entry:
    bay_id: str
    reason: str
    priority: int
    added: datetime
    not_before: datetime | None = None


@dataclass
class MissionQueue:
    cfg: PlannerCfg
    entries: dict[str, Entry] = field(default_factory=dict)
    _visited: dict[str, datetime] = field(default_factory=dict)

    def add(
        self, bay_id: str, reason: str, t: datetime, not_before: datetime | None = None
    ) -> bool:
        """Queue a bay; returns True if it was not queued before."""
        last = self._visited.get(bay_id)
        cooling = last and t - last < timedelta(minutes=self.cfg.requeue_cooldown_min)
        if cooling and reason in self.cfg.cooldown_reasons:
            return False
        prio = self.cfg.reason_priority.get(reason, 0)
        old = self.entries.get(bay_id)
        if old is None:
            self.entries[bay_id] = Entry(bay_id, reason, prio, t, not_before)
            return True
        if prio > old.priority:
            old.reason, old.priority = reason, prio
        return False

    def drop(self, bay_id: str, reason: str | None = None) -> None:
        """Forget a bay (only if its reason matches, when given)."""
        e = self.entries.get(bay_id)
        if e and (reason is None or e.reason == reason):
            del self.entries[bay_id]

    def visited(self, bay_id: str, t: datetime) -> None:
        self.entries.pop(bay_id, None)
        self._visited[bay_id] = t

    def _due(self, e: Entry, now: datetime) -> bool:
        if e.not_before and now < e.not_before:
            return False
        waited = now - (e.not_before or e.added)
        return e.priority >= self.cfg.ready_priority or waited >= timedelta(
            minutes=self.cfg.ready_after_min
        )

    def ready(self, now: datetime) -> bool:
        return any(self._due(e, now) for e in self.entries.values())

    def pop(self, n: int, now: datetime) -> list[Entry]:
        """Up to n due entries, most urgent (then oldest) first."""
        due = sorted((e for e in self.entries.values() if self._due(e, now)),
                     key=lambda e: (-e.priority, e.added, e.bay_id))[:n]  # fmt: skip
        for e in due:
            del self.entries[e.bay_id]
        return due
