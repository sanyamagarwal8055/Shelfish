"""The Vision <-> Brain interface contract, as code. Mirrors docs/CONTRACT.md.

Change this file only through a PR labelled `contract`, with a CONTRACT_VERSION bump and the
other person's approval. Both tracks import from here; neither imports the other's internals.

Every model forbids unknown fields, so a typo in a producer fails loudly at the consumer.
Use the `parse_*` helpers on dicts loaded from JSON; use `to_json_dict` to write them back.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONTRACT_VERSION = "1.0"

# Fixed by the contract (not tunable): shelf rows, bay width, trust threshold.
ROWS = range(6)  # 0 = base deck ... 5 = top
BAY_WIDTH_CM = 120.0
MIN_GAP_CM = 5.0
QUALITY_MIN_TRUSTED = 0.5

BAY_ID_RE = re.compile(r"^G(10|[1-9])-(?:(L|R)-(0[1-9]|10)|E-(F|B))$")
SKU_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]*$")
UNKNOWN_SKU = "UNKNOWN"
AMBIGUOUS_PREFIX = "AMBIGUOUS:"

_SKU = r"[A-Z0-9][A-Z0-9_]*"
_BAY = r"G(10|[1-9])-(?:(L|R)-(0[1-9]|10)|E-(F|B))"

BayId = Annotated[str, StringConstraints(pattern=rf"^{_BAY}$")]
SkuId = Annotated[str, StringConstraints(pattern=rf"^{_SKU}$")]
# A real sku_id, "AMBIGUOUS:<a>|<b>[|<c>...]", or "UNKNOWN" (which also matches the sku_id form).
SkuRef = Annotated[str, StringConstraints(pattern=rf"^(?:{_SKU}|AMBIGUOUS:{_SKU}(?:\|{_SKU})+)$")]
Cm = Annotated[float, Field(ge=0.0, le=BAY_WIDTH_CM)]
Source = Literal["camera", "robot"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------
# IDs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BayIdParts:
    run: str  # "G1".."G10"
    face: str  # "L", "R" or "E" (end cap)
    index: int | None  # 1..10 for aisle bays, None for end caps
    end: str | None  # "F" (front) or "B" (back) for end caps, else None

    @property
    def is_end_cap(self) -> bool:
        return self.face == "E"


def parse_bay_id(bay_id: str) -> BayIdParts:
    """Split a bay_id into its parts. Raises ValueError if it is not a contract bay_id."""
    m = BAY_ID_RE.match(bay_id)
    if not m:
        raise ValueError(f"invalid bay_id {bay_id!r}")
    run = f"G{m.group(1)}"
    if m.group(4):
        return BayIdParts(run=run, face="E", index=None, end=m.group(4))
    return BayIdParts(run=run, face=m.group(2), index=int(m.group(3)), end=None)


def is_ambiguous(sku: str) -> bool:
    return sku.startswith(AMBIGUOUS_PREFIX)


def ambiguous_candidates(sku: str) -> list[str]:
    """`AMBIGUOUS:A|B` -> ["A", "B"]; a plain sku -> [sku]."""
    if is_ambiguous(sku):
        return sku[len(AMBIGUOUS_PREFIX) :].split("|")
    return [sku]


# --------------------------------------------------------------------------------------------
# BayReading (Vision -> Brain)
# --------------------------------------------------------------------------------------------


class Pack(_Model):
    sku: SkuRef
    conf: float = Field(ge=0.0, le=1.0)
    x_cm: Cm  # left edge
    w_cm: float = Field(gt=0.0)
    h_cm: float = Field(gt=0.0)
    stack: int = Field(ge=1)
    depth_left: int | None = Field(ge=0)  # required key; null = depth unknown


class Gap(_Model):
    x_cm: Cm
    w_cm: float = Field(gt=MIN_GAP_CM)

    @model_validator(mode="after")
    def _inside_bay(self) -> Gap:
        if self.x_cm + self.w_cm > BAY_WIDTH_CM + 1e-6:
            raise ValueError(f"gap {self.x_cm}+{self.w_cm} runs past the bay edge")
        return self


class Label(_Model):
    x_cm: Cm
    sku: SkuId
    price: float = Field(ge=0.0)


class Row(_Model):
    row: int = Field(ge=ROWS.start, le=ROWS.stop - 1)
    occluded: list[list[float]] = Field(default_factory=list)
    packs: list[Pack] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_occluded(self) -> Row:
        for r in self.occluded:
            if len(r) != 2 or not (0.0 <= r[0] < r[1] <= BAY_WIDTH_CM):
                raise ValueError(f"occluded range {r} must be [a, b] with 0 <= a < b <= 120")
        return self


class BayReading(_Model):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    bay_id: BayId
    source: Source
    t: AwareDatetime
    frame_ref: str
    quality: float = Field(ge=0.0, le=1.0)
    px_per_cm: float = Field(gt=0.0)
    rows: list[Row]

    @model_validator(mode="after")
    def _check(self) -> BayReading:
        seen = [r.row for r in self.rows]
        if len(seen) != len(set(seen)):
            raise ValueError(f"duplicate row numbers in {seen}")
        if self.source == "camera" and any(r.labels for r in self.rows):
            raise ValueError("labels must be [] for camera readings")
        return self


def is_trusted(reading: BayReading) -> bool:
    """Contract rule: quality below 0.5 means the bay counts as unseen."""
    return reading.quality >= QUALITY_MIN_TRUSTED


# --------------------------------------------------------------------------------------------
# Mission (Brain -> robot bridge) and RobotStatus (robot bridge -> Brain)
# --------------------------------------------------------------------------------------------


class Mission(_Model):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    mission_id: Annotated[str, StringConstraints(pattern=r"^M-\d+$")]
    kind: Literal["sweep", "mission"]
    created_at: AwareDatetime
    bays: list[BayId]  # visit order; may be empty for a sweep (= all bays, fixed route)
    # Free text in v1.0. Reasons in use: blocked, verify, conflict, ambiguous, promo.
    reasons: dict[BayId, Annotated[str, StringConstraints(min_length=1)]] = Field(
        default_factory=dict
    )
    speed_mps: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> Mission:
        if self.kind == "mission" and not self.bays:
            raise ValueError("a mission needs at least one bay")
        if len(self.bays) != len(set(self.bays)):
            raise ValueError("duplicate bays in mission")
        extra = set(self.reasons) - set(self.bays)
        if extra:
            raise ValueError(f"reasons for bays not in the mission: {sorted(extra)}")
        return self


class RobotStatus(_Model):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    t: AwareDatetime
    state: Literal["idle", "docked", "running", "blocked"]
    x_m: float
    y_m: float
    mission_id: str | None = None
    done_bays: list[BayId] = Field(default_factory=list)
    skipped_bays: list[BayId] = Field(default_factory=list)
    pending_bays: list[BayId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _disjoint(self) -> RobotStatus:
        groups = [self.done_bays, self.skipped_bays, self.pending_bays]
        total = sum(len(g) for g in groups)
        if len(set().union(*groups)) != total:
            raise ValueError("a bay appears twice across done/skipped/pending")
        return self


# --------------------------------------------------------------------------------------------
# Planogram / label map (Brain writes, Vision reads as a hint)
# --------------------------------------------------------------------------------------------


class PlanogramSlot(_Model):
    position: int = Field(ge=0)
    sku_id: SkuId
    x_start_cm: Cm
    x_end_cm: Cm
    facings: int = Field(ge=0)
    min_facings: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> PlanogramSlot:
        if self.x_end_cm <= self.x_start_cm:
            raise ValueError("x_end_cm must be greater than x_start_cm")
        if self.min_facings > self.facings:
            raise ValueError("min_facings cannot exceed facings")
        return self


class Planogram(_Model):
    bay_id: BayId
    source: Literal["planogram", "label_map"] = "planogram"
    updated: AwareDatetime | None = None
    rows: list[list[PlanogramSlot]]  # rows[i] is shelf row i

    @model_validator(mode="after")
    def _check(self) -> Planogram:
        if len(self.rows) > len(ROWS):
            raise ValueError(f"at most {len(ROWS)} rows")
        for i, row in enumerate(self.rows):
            for a, b in zip(row, row[1:], strict=False):
                if b.x_start_cm < a.x_end_cm - 1e-6:
                    raise ValueError(f"row {i}: slots overlap or are out of order")
        return self

    def skus(self) -> set[str]:
        return {s.sku_id for row in self.rows for s in row}


# --------------------------------------------------------------------------------------------
# Shared read-only data: sku_master.csv
# --------------------------------------------------------------------------------------------

SKU_MASTER_COLUMNS = [
    "sku_id",
    "name",
    "category",
    "width_cm",
    "height_cm",
    "depth_cm",
    "margin_inr",
    "pack_keywords",
    "home_bay",
]


class SkuRow(_Model):
    sku_id: SkuId
    name: str
    category: str
    width_cm: float = Field(gt=0.0)
    height_cm: float = Field(gt=0.0)
    depth_cm: float = Field(gt=0.0)
    margin_inr: float = Field(ge=0.0)
    pack_keywords: list[str]  # ";"-separated in the CSV
    home_bay: BayId


def load_sku_master(path: str | Path = "data/sku_master.csv") -> dict[str, SkuRow]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SKU_MASTER_COLUMNS:
            raise ValueError(f"sku_master columns must be {SKU_MASTER_COLUMNS}")
        out: dict[str, SkuRow] = {}
        for rec in reader:
            kw = [k.strip() for k in rec["pack_keywords"].split(";") if k.strip()]
            row = SkuRow.model_validate({**rec, "pack_keywords": kw})
            if row.sku_id in out:
                raise ValueError(f"duplicate sku_id {row.sku_id}")
            out[row.sku_id] = row
    return out


def check_skus(skus: Iterable[str], known: Iterable[str]) -> list[str]:
    """Return SKU references (real or inside AMBIGUOUS:) that are not in `known`."""
    known = set(known)
    missing = []
    for sku in skus:
        if sku == UNKNOWN_SKU:
            continue
        missing += [c for c in ambiguous_candidates(sku) if c not in known]
    return sorted(set(missing))


def reading_skus(reading: BayReading) -> set[str]:
    """Every sku string mentioned in a reading (packs and labels)."""
    return {p.sku for r in reading.rows for p in r.packs} | {
        lb.sku for r in reading.rows for lb in r.labels
    }


# --------------------------------------------------------------------------------------------
# Parse / dump helpers
# --------------------------------------------------------------------------------------------


def parse_bay_reading(obj: dict) -> BayReading:
    return BayReading.model_validate(obj)


def parse_mission(obj: dict) -> Mission:
    return Mission.model_validate(obj)


def parse_robot_status(obj: dict) -> RobotStatus:
    return RobotStatus.model_validate(obj)


def parse_planogram(obj: dict) -> Planogram:
    return Planogram.model_validate(obj)


def to_json_dict(model: BaseModel) -> dict:
    """Model -> plain JSON-ready dict in contract form (ISO times, nulls kept)."""
    return model.model_dump(mode="json")
