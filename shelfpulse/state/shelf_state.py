"""Camera + robot fusion per slot: spot when the two sources disagree.

Each source's latest verdict per slot is kept. If a new verdict says "bad" (OUT/LOW) while the
other source said "fine" (OK) within `conflict_window_s`, or the other way round, the slot is
UNSURE: the reading gets no vote in the alert tracker and the bay is queued for a re-check.
Otherwise the newest verdict simply votes, so the newest confident reading wins.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime

BAD = ("OUT", "LOW")


@dataclass
class ShelfState:
    conflict_window_s: float
    _last: dict[tuple[Hashable, str], tuple[bool, datetime]] = field(default_factory=dict)

    def conflict(self, key: Hashable, source: str, status: str, t: datetime) -> bool:
        """Record this verdict; True if it contradicts the other source's recent one."""
        bad = status in BAD
        other = "robot" if source == "camera" else "camera"
        prev = self._last.get((key, other))
        self._last[(key, source)] = (bad, t)
        if prev is None:
            return False
        prev_bad, prev_t = prev
        return prev_bad != bad and abs((t - prev_t).total_seconds()) <= self.conflict_window_s
