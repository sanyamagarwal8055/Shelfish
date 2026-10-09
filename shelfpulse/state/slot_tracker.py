"""k-of-n alert state machine, one track per key (a slot, or a stray pack in a slot).

Each trusted frame reports a key as bad (a kind such as "OUT") or OK (None); unseen keys are
simply not reported. An alert is raised when k of the last n frames are bad, and resolved after
`clear_after` OK frames in a row. One noisy frame therefore raises nothing.

Kinds have a severity order (OUT before LOW): the alert takes the most severe kind that at least
k frames reach, counting more severe frames too, so OUT + LOW gives LOW and OUT + OUT gives OUT.
A running alert that changes kind (LOW -> OUT) emits a new transition.

A robot frame is high resolution and may be the only look a bay gets for hours, so with
`robot_overrides` it fills the whole window: one robot frame raises or resolves on its own.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime

SEVERITY = ("OUT", "LOW", "LOW_ESTIMATED", "MISPLACED", "UNKNOWN_ITEM")  # most severe first
RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class Transition:
    key: Hashable
    kind: str  # an alert kind, or RESOLVED
    previous: str | None  # alert kind before this transition
    since: datetime  # first bad frame of the alert (for RESOLVED: when it was raised)
    t: datetime


@dataclass
class _Track:
    window: deque[tuple[str | None, datetime]]
    active: str | None = None
    since: datetime | None = None
    ok_run: int = 0


@dataclass
class SlotTracker:
    k: int
    n: int
    clear_after: int
    robot_overrides: bool = True
    _tracks: dict[Hashable, _Track] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.k <= self.n:
            raise ValueError("need 1 <= k <= n")

    def active(self, key: Hashable) -> str | None:
        tr = self._tracks.get(key)
        return tr.active if tr else None

    def quiet(self, key: Hashable) -> bool:
        """No alert and no bad frame left in the window: nothing to keep watching."""
        tr = self._tracks.get(key)
        return tr is None or (tr.active is None and all(k is None for k, _ in tr.window))

    def update(
        self, key: Hashable, kind: str | None, t: datetime, source: str = "camera"
    ) -> Transition | None:
        if kind is not None and kind not in SEVERITY:
            raise ValueError(f"unknown alert kind {kind!r}")
        tr = self._tracks.setdefault(key, _Track(window=deque(maxlen=self.n)))
        if source == "robot" and self.robot_overrides:
            tr.window.clear()
            tr.window.extend([(kind, t)] * self.n)
            tr.ok_run = self.clear_after if kind is None else 0
        else:
            tr.window.append((kind, t))
            tr.ok_run = tr.ok_run + 1 if kind is None else 0

        if tr.active is not None and tr.ok_run >= self.clear_after:
            out = Transition(key, RESOLVED, tr.active, tr.since, t)
            tr.window.clear()
            tr.active, tr.since = None, None
            return out

        level = self._level(tr)
        if level is None or level == tr.active:
            return None
        previous = tr.active
        if previous is None:
            tr.since = next(ts for k, ts in tr.window if k is not None)
        tr.active = level
        return Transition(key, level, previous, tr.since, t)

    def _level(self, tr: _Track) -> str | None:
        ranks = [SEVERITY.index(k) for k, _ in tr.window if k is not None]
        for rank, kind in enumerate(SEVERITY):
            if sum(r <= rank for r in ranks) >= self.k:
                return kind
        return None
