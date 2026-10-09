"""Planograms rebuilt from the shelf-edge labels the robot reads, and drift from the digital one.

Robot readings carry, per row, the price labels on the shelf edge: {x_cm, sku, price}. Each
label owns the shelf from its x to the next label's x (the last one to the bay edge); a first
label within `snap_start_cm` of the left edge owns from 0. Facings = floor(width / pack width)
from sku_master, min_facings = ceil(facings x min_facings_frac). A row with no labels in a
reading keeps what was known, and so does a row read with fewer labels than known when every
label read matches a known one (same SKU, start within drift_tol_cm): OCR missed a tag, the
products didn't change. A new or moved label still updates the row. The result is a contract
Planogram with source "label_map", written as <dir>/<bay_id>.json for the Vision track to use
as a location hint, and used by the matcher for bays that have no digital planogram.

Drift: where both exist, a label map row that disagrees with the digital planogram (a slot whose
SKU differs, or whose edges moved by more than drift_tol_cm) is reported once it has persisted
for drift_days, so a label that was briefly moved is not flagged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    BayReading,
    Planogram,
    PlanogramSlot,
    SkuRow,
    to_json_dict,
)


class LabelMapCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snap_start_cm: float = Field(ge=0)
    min_facings_frac: float = Field(gt=0, le=1)
    drift_tol_cm: float = Field(ge=0)
    drift_days: float = Field(ge=0)


class Drift(BaseModel):
    """One persistent difference between the label map and the digital planogram."""

    model_config = ConfigDict(extra="forbid")
    bay_id: str
    row: int
    x_cm: float  # where on the shelf (planogram slot start, or label start if unplanned)
    planned: str | None  # SKU in the digital planogram there
    labelled: str | None  # SKU on the shelf label there
    kind: str  # "sku" (different product) or "moved" (slot edges shifted)
    since: datetime
    t: datetime


def rows_from_labels(
    reading: BayReading, skus: dict[str, SkuRow], cfg: LabelMapCfg
) -> dict[int, list[PlanogramSlot]]:
    """Slots per row for every row of a robot reading that has labels."""
    out = {}
    for row in reading.rows:
        labels = sorted((lb for lb in row.labels if lb.sku in skus), key=lambda lb: lb.x_cm)
        if not labels:
            continue
        slots = []
        for i, lb in enumerate(labels):
            start = 0.0 if i == 0 and lb.x_cm <= cfg.snap_start_cm else lb.x_cm
            end = labels[i + 1].x_cm if i + 1 < len(labels) else BAY_WIDTH_CM
            if end - start <= 0:
                continue
            facings = max(1, int((end - start) // skus[lb.sku].width_cm))
            slots.append(
                PlanogramSlot(
                    position=len(slots),
                    sku_id=lb.sku,
                    x_start_cm=round(start, 1),
                    x_end_cm=round(end, 1),
                    facings=facings,
                    min_facings=max(1, math.ceil(facings * cfg.min_facings_frac)),
                )
            )
        out[row.row] = slots
    return out


def drift_rows(plan: Planogram, labels: Planogram, tol_cm: float) -> list[tuple]:
    """(row, x_cm, planned, labelled, kind) for every disagreement right now."""
    diffs = []
    for r in range(max(len(plan.rows), len(labels.rows))):
        planned = plan.rows[r] if r < len(plan.rows) else []
        seen = labels.rows[r] if r < len(labels.rows) else []
        if not seen:  # no labels read on this row: nothing to compare
            continue
        for s in planned:
            mid = (s.x_start_cm + s.x_end_cm) / 2
            match = next((x for x in seen if x.x_start_cm <= mid < x.x_end_cm), None)
            if match is None or match.sku_id != s.sku_id:
                diffs.append((r, s.x_start_cm, s.sku_id, match.sku_id if match else None, "sku"))
            elif (abs(match.x_start_cm - s.x_start_cm) > tol_cm
                  or abs(match.x_end_cm - s.x_end_cm) > tol_cm):  # fmt: skip
                diffs.append((r, s.x_start_cm, s.sku_id, match.sku_id, "moved"))
        for x in seen:  # labels for products the planogram doesn't have on this row
            if all(s.sku_id != x.sku_id for s in planned):
                diffs.append((r, x.x_start_cm, None, x.sku_id, "sku"))
    return sorted(set(diffs), key=lambda d: (d[0], d[1], d[4], str(d[3])))


@dataclass
class LabelMaps:
    skus: dict[str, SkuRow]
    cfg: LabelMapCfg
    maps: dict[str, Planogram] = field(default_factory=dict)
    _first_seen: dict[tuple, datetime] = field(default_factory=dict)
    _reported: set[tuple] = field(default_factory=set)

    def update(self, reading: BayReading) -> Planogram | None:
        """Fold a robot reading's labels into the bay's label map; returns it if it changed."""
        if reading.source != "robot":
            return None
        new_rows = rows_from_labels(reading, self.skus, self.cfg)
        if not new_rows:
            return None
        old = self.maps.get(reading.bay_id)
        rows = [list(r) for r in old.rows] if old else []
        while len(rows) <= max(new_rows):
            rows.append([])
        for r, slots in new_rows.items():
            if not self._missed_tags(rows[r], slots):
                rows[r] = slots
        plan = Planogram(bay_id=reading.bay_id, source="label_map", updated=reading.t, rows=rows)
        if old is not None and old.rows == plan.rows:
            self.maps[reading.bay_id] = plan  # same layout, newer timestamp
            return None
        self.maps[reading.bay_id] = plan
        return plan

    def _missed_tags(self, known: list[PlanogramSlot], seen: list[PlanogramSlot]) -> bool:
        """Fewer labels than known, all of them known ones: unread tags, not a new layout."""
        tol = self.cfg.drift_tol_cm
        return len(seen) < len(known) and all(
            any(k.sku_id == s.sku_id and abs(k.x_start_cm - s.x_start_cm) <= tol for k in known)
            for s in seen
        )

    def drift(self, digital: Planogram, t: datetime) -> list[Drift]:
        """Disagreements with the digital planogram that just reached drift_days."""
        lm = self.maps.get(digital.bay_id)
        if lm is None:
            return []
        now = {(digital.bay_id, *d) for d in drift_rows(digital, lm, self.cfg.drift_tol_cm)}
        for gone in [k for k in self._first_seen if k[0] == digital.bay_id and k not in now]:
            del self._first_seen[gone]  # fixed: forget it (and report again if it comes back)
            self._reported.discard(gone)
        out = []
        for key in sorted(now, key=str):
            since = self._first_seen.setdefault(key, t)
            if key not in self._reported and t - since >= timedelta(days=self.cfg.drift_days):
                self._reported.add(key)
                bay, row, x, planned, labelled, kind = key
                out.append(Drift(bay_id=bay, row=row, x_cm=x, planned=planned, labelled=labelled,
                                 kind=kind, since=since, t=t))  # fmt: skip
        return out


def write(plan: Planogram, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{plan.bay_id}.json"
    path.write_text(json.dumps(to_json_dict(plan)) + "\n", encoding="utf-8")
    return path
