"""The simulated store: true shelf contents, stock and POS, and the readings Vision would send.

Each planogram slot holds `facings` columns of packs, front to back, each up to `depth_cap`
deep (shelf depth / pack depth). A column with at least one pack shows as one facing, and its
count is the depth_left a stereo camera or the robot's ToF would measure. Shoppers and thieves
take from a random non-empty column; staff fill columns round-robin so facings reappear first.

Stock follows the usual store-system rules: a sale lowers the system on-hand count, a theft
does not, and a restock only moves packs from the backroom to the shelf.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    MIN_GAP_CM,
    BayReading,
    Gap,
    Label,
    Pack,
    Planogram,
    Row,
    SkuRow,
)
from shelfpulse.decision.store_data import StoreData
from sim.scenario import Scenario, SimEvent


@dataclass
class Slot:
    bay_id: str
    row: int
    position: int
    sku: str
    x_start: float
    width: float  # pack width, cm
    depth_cap: int
    cols: list[int]  # packs per facing column, left to right

    @property
    def units(self) -> int:
        return sum(self.cols)

    @property
    def capacity(self) -> int:
        return len(self.cols) * self.depth_cap

    def add(self, n: int) -> int:
        """Fill round-robin, front facings first. Returns how many packs fitted."""
        added = 0
        while added < n and self.units < self.capacity:
            i = min(range(len(self.cols)), key=lambda j: (self.cols[j], j))
            self.cols[i] += 1
            added += 1
        return added

    def take(self, n: int, rng: np.random.Generator) -> int:
        taken = 0
        while taken < n and self.units:
            live = [i for i, c in enumerate(self.cols) if c]
            self.cols[live[int(rng.integers(len(live)))]] -= 1
            taken += 1
        return taken


def spread(units: int, facings: int, depth_cap: int) -> list[int]:
    """`units` packs spread round-robin over `facings` columns (what a reading shows)."""
    s = Slot("", 0, 0, "", 0.0, 1.0, depth_cap, [0] * facings)
    s.add(units)
    return s.cols


@dataclass
class Window:
    until: datetime
    data: object


@dataclass
class SimStore:
    sc: Scenario
    skus: dict[str, SkuRow]
    plans: dict[str, Planogram]
    shelf_depth_cm: float
    rng: np.random.Generator
    slots: dict[tuple[str, int, int], Slot] = field(default_factory=dict)
    strays: dict[tuple[str, int], list[tuple[str, float]]] = field(default_factory=dict)
    backroom: dict[str, int] = field(default_factory=dict)
    system: dict[str, int] = field(default_factory=dict)
    pos: list[tuple[datetime, str, int]] = field(default_factory=list)
    inventory: list[tuple[datetime, str, int, int]] = field(default_factory=list)
    lost_sales: dict[str, int] = field(default_factory=dict)
    occlusions: dict[str, list[Window]] = field(default_factory=dict)
    quality: dict[str, Window] = field(default_factory=dict)
    misreads: dict[tuple[str, str, int, int], Window] = field(default_factory=dict)
    ambiguous: dict[str, Window] = field(default_factory=dict)
    unidentified: dict[str, Window] = field(default_factory=dict)
    data: StoreData = field(default_factory=StoreData)  # live POS + stock, as the Brain sees it
    robot_log: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # mission -> bays
    dock_xy: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def build(cls, sc: Scenario, skus: dict[str, SkuRow], plans: dict[str, Planogram],
              shelf_depth_cm: float) -> SimStore:  # fmt: skip
        st = cls(sc, skus, plans, shelf_depth_cm, np.random.default_rng(sc.seed))
        st.data.data_start = sc.t_start
        for bay in sc.bays:
            for r, row in enumerate(plans[bay].rows):
                for s in row:
                    sku = skus[s.sku_id]
                    cap = max(1, int(shelf_depth_cm // sku.depth_cm))
                    slot = Slot(bay, r, s.position, s.sku_id, s.x_start_cm, sku.width_cm, cap,
                                [0] * s.facings)  # fmt: skip
                    slot.add(round(sc.fill * slot.capacity))
                    st.slots[(bay, r, s.position)] = slot
        for u in sc.start_units:
            slot = st.slots.get((u.bay, u.row, u.position))
            if slot is None:
                raise ValueError(f"start_units: no slot {u.bay} row {u.row} #{u.position}")
            if u.units > slot.capacity:
                raise ValueError(f"start_units: {u.units} > capacity {slot.capacity} of {slot.sku}")
            slot.cols = spread(u.units, len(slot.cols), slot.depth_cap)
        on_shelf: dict[str, int] = {}
        for s in st.slots.values():
            on_shelf[s.sku] = on_shelf.get(s.sku, 0) + s.units
        for sku in sorted(set(on_shelf) | set(sc.stock)):
            stock = sc.stock.get(sku)
            st.backroom[sku] = stock.backroom if stock else 0
            extra = stock.system_extra if stock else 0
            st.system[sku] = on_shelf.get(sku, 0) + st.backroom[sku] + extra
            st.snapshot(sc.t_start, sku)
        return st

    # --- stock bookkeeping ---------------------------------------------------------------------

    def snapshot(self, t: datetime, sku: str) -> None:
        row = (t, sku, self.system.get(sku, 0), self.backroom.get(sku, 0))
        self.inventory.append(row)
        self.data.snapshots.setdefault(sku, []).append((t, row[2], row[3]))

    def sell(self, t: datetime, slot: Slot, qty: int) -> None:
        sold = slot.take(qty, self.rng)
        if sold:
            self.pos.append((t, slot.sku, sold))
            self.data.sales.setdefault(slot.sku, []).append((t, sold))
            self.system[slot.sku] = self.system.get(slot.sku, 0) - sold
        if qty > sold:
            self.lost_sales[slot.sku] = self.lost_sales.get(slot.sku, 0) + qty - sold

    def background_sales(self, t: datetime) -> None:
        for sku, per_hour in sorted(self.sc.sales_per_hour.items()):
            n = int(self.rng.poisson(per_hour / 60.0))
            for _ in range(n):
                live = [s for s in self.slots.values() if s.sku == sku and s.units]
                if not live:
                    self.lost_sales[sku] = self.lost_sales.get(sku, 0) + 1
                    continue
                self.sell(t, live[int(self.rng.integers(len(live)))], 1)

    # --- events --------------------------------------------------------------------------------

    def apply(self, ev: SimEvent, t: datetime) -> None:
        until = self.sc.until(ev)
        slot = self.slots.get((ev.bay, ev.row, ev.position)) if ev.position is not None else None
        if ev.do in ("sell", "theft", "restock", "face_up", "misread") and slot is None:
            raise ValueError(f"{ev.do} at {ev.at}: no slot {ev.bay} row {ev.row} #{ev.position}")
        if ev.do == "sell":
            self.sell(t, slot, ev.qty)
        elif ev.do == "theft":
            slot.take(ev.qty, self.rng)
        elif ev.do == "restock":
            want = slot.capacity - slot.units if ev.qty is None else ev.qty
            moved = slot.add(min(want, self.backroom.get(slot.sku, 0)))
            self.backroom[slot.sku] = self.backroom.get(slot.sku, 0) - moved
            self.snapshot(t, slot.sku)
        elif ev.do == "face_up":
            slot.cols = spread(slot.units, len(slot.cols), slot.depth_cap)  # same packs, spread
        elif ev.do == "misplace":
            self.strays.setdefault((ev.bay, ev.row), []).append((ev.sku, ev.x_cm))
        elif ev.do == "unmisplace":
            self.strays.pop((ev.bay, ev.row), None)
        elif ev.do == "occlude":
            rows = tuple(ev.rows) if ev.rows is not None else None
            self.occlusions.setdefault(ev.bay, []).append(Window(until, (ev.x, rows, ev.source)))
        elif ev.do == "quality":
            self.quality[ev.bay] = Window(until, ev.value)
        elif ev.do == "misread":
            self.misreads[(ev.source, ev.bay, ev.row, ev.position)] = Window(until, ev.units)
        elif ev.do == "ambiguous":
            self.ambiguous[ev.sku] = Window(until, ev.candidates)
        elif ev.do == "unidentified":
            self.unidentified[ev.sku] = Window(until, None)
        elif ev.do == "set_system":
            self.system[ev.sku] = ev.qty
            self.snapshot(t, ev.sku)

    # --- what Vision would report ---------------------------------------------------------------

    def reading(self, bay: str, source: str, t: datetime, quality: float, conf: float,
                ambiguous_conf: float) -> BayReading:  # fmt: skip
        occ_windows = [w for w in self.occlusions.get(bay, [])
                       if t < w.until and w.data[2] in (None, source)]  # fmt: skip
        q = self.quality.get(bay)
        if source == "camera" and q and t < q.until:
            quality = q.data
        rows = []
        for r, plan_row in enumerate(self.plans[bay].rows):
            occluded = sorted(
                list(w.data[0]) for w in occ_windows if w.data[1] is None or r in w.data[1]
            )
            packs = []
            for s in plan_row:
                slot = self.slots[(bay, r, s.position)]
                packs += self._slot_packs(slot, source, t, conf, ambiguous_conf)
            for sku, x in self.strays.get((bay, r), []):
                packs.append(self._pack(sku, x, 1, conf, ambiguous_conf, t, source))
            packs = [p for p in packs if not _hidden(p, occluded)]
            packs.sort(key=lambda p: p.x_cm)
            labels = []
            if source == "robot":
                labels = [Label(x_cm=s.x_start_cm, sku=s.sku_id, price=self._price(s.sku_id))
                          for s in plan_row]  # fmt: skip
            rows.append(Row(row=r, occluded=occluded, packs=packs,
                            gaps=_gaps(packs, occluded), labels=labels))  # fmt: skip
        return BayReading(
            bay_id=bay,
            source=source,
            t=t,
            frame_ref=f"sim/{bay}/{source}/{t:%Y-%m-%dT%H-%M}.jpg",
            quality=quality,
            px_per_cm=10.0,
            rows=rows,
        )

    def _slot_packs(self, slot: Slot, source: str, t: datetime, conf: float,
                    ambiguous_conf: float) -> list[Pack]:  # fmt: skip
        cols = slot.cols
        mis = self.misreads.get((source, slot.bay_id, slot.row, slot.position))
        if mis and t < mis.until:
            cols = spread(mis.data, len(cols), slot.depth_cap)
        return [
            self._pack(slot.sku, slot.x_start + i * slot.width, c, conf, ambiguous_conf, t, source)
            for i, c in enumerate(cols)
            if c
        ]

    def _pack(self, sku: str, x: float, depth: int, conf: float, ambiguous_conf: float,
              t: datetime, source: str) -> Pack:  # fmt: skip
        row = self.skus[sku]
        name, c = sku, conf
        amb = self.ambiguous.get(sku)
        if amb and t < amb.until:
            name, c = "AMBIGUOUS:" + "|".join(amb.data), ambiguous_conf
        unk = self.unidentified.get(sku)
        if unk and t < unk.until:
            name, c = "UNKNOWN", ambiguous_conf
        has_depth = source == "robot" or self.sc.depth_known  # robot ToF always measures depth
        return Pack(sku=name, conf=c, x_cm=round(x, 2), w_cm=row.width_cm, h_cm=row.height_cm,
                    stack=1, depth_left=depth if has_depth else None)  # fmt: skip

    def _price(self, sku: str) -> float:
        return round(self.skus[sku].margin_inr * 5, 2)  # illustrative shelf price

    # --- outputs -------------------------------------------------------------------------------

    def write_csvs(self, out: Path) -> None:
        with open(out / "pos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "sku_id", "qty"])
            w.writerows((t.isoformat(), sku, q) for t, sku, q in self.pos)
        with open(out / "inventory.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "sku_id", "system_on_hand", "backroom"])
            w.writerows((t.isoformat(), sku, s, b) for t, sku, s, b in self.inventory)


def _hidden(p: Pack, occluded: list[list[float]]) -> bool:
    centre = p.x_cm + p.w_cm / 2
    return any(a <= centre <= b for a, b in occluded)


def _gaps(packs: list[Pack], occluded: list[list[float]]) -> list[Gap]:
    """Visible empty stretches wider than MIN_GAP_CM (occluded ranges are not 'empty')."""
    blocked = sorted([(p.x_cm, min(p.x_cm + p.w_cm, BAY_WIDTH_CM)) for p in packs] +
                     [(a, b) for a, b in occluded])  # fmt: skip
    gaps, cursor = [], 0.0
    for a, b in blocked:
        if a - cursor > MIN_GAP_CM:
            gaps.append(Gap(x_cm=round(cursor, 2), w_cm=round(a - cursor, 2)))
        cursor = max(cursor, b)
    if BAY_WIDTH_CM - cursor > MIN_GAP_CM:
        gaps.append(Gap(x_cm=round(cursor, 2), w_cm=round(BAY_WIDTH_CM - cursor, 2)))
    return gaps
