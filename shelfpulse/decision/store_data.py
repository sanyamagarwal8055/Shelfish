"""Store data the diagnosis needs: POS sales and stock counts.

Phase 1 reads the simulator's (or a store export's) CSVs:

    pos.csv        t, sku_id, qty                          one line per sale
    inventory.csv  t, sku_id, system_on_hand, backroom     a snapshot whenever stock changes
                                                           other than by a sale

The system on-hand count at time t is the last snapshot before t minus the sales since then
(a sale lowers it; a theft does not, which is what makes phantom stock).
"""

from __future__ import annotations

import bisect
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class StoreData:
    sales: dict[str, list[tuple[datetime, int]]] = field(default_factory=dict)
    snapshots: dict[str, list[tuple[datetime, int, int]]] = field(default_factory=dict)
    data_start: datetime | None = None

    @classmethod
    def from_csv(cls, pos: Path, inventory: Path) -> StoreData:
        sales: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
        snaps: dict[str, list[tuple[datetime, int, int]]] = defaultdict(list)
        with open(pos, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sales[r["sku_id"]].append((_aware(r["t"]), int(r["qty"])))
        with open(inventory, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                snaps[r["sku_id"]].append(
                    (_aware(r["t"]), int(r["system_on_hand"]), int(r["backroom"]))
                )
        for v in (*sales.values(), *snaps.values()):
            v.sort(key=lambda x: x[0])
        times = [v[0][0] for v in (*sales.values(), *snaps.values()) if v]
        return cls(dict(sales), dict(snaps), min(times) if times else None)

    def sold(self, sku: str, t0: datetime, t1: datetime) -> int:
        """Packs sold in (t0, t1]."""
        rows = self.sales.get(sku, [])
        lo = bisect.bisect_right(rows, t0, key=lambda x: x[0])
        hi = bisect.bisect_right(rows, t1, key=lambda x: x[0])
        return sum(q for _, q in rows[lo:hi])

    def _snapshot(self, sku: str, t: datetime) -> tuple[datetime, int, int] | None:
        rows = self.snapshots.get(sku, [])
        i = bisect.bisect_right(rows, t, key=lambda x: x[0])
        return rows[i - 1] if i else None

    def system_on_hand(self, sku: str, t: datetime) -> int | None:
        snap = self._snapshot(sku, t)
        if snap is None:
            return None
        ts, on_hand, _ = snap
        return on_hand - self.sold(sku, ts, t)

    def backroom(self, sku: str, t: datetime) -> int | None:
        snap = self._snapshot(sku, t)
        return snap[2] if snap else None

    def velocity(self, sku: str, t: datetime, window_h: float, min_span_h: float) -> float:
        """Packs per hour over the trailing window (or since the data starts, at least
        min_span_h, so a short history doesn't inflate the rate)."""
        if self.data_start is None:
            return 0.0
        span_h = (t - self.data_start).total_seconds() / 3600
        hours = max(min_span_h, min(window_h, span_h))
        return self.sold(sku, t - timedelta(hours=hours), t) / hours


def _aware(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise ValueError(f"time {s!r} has no timezone")
    return t
