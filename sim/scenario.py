"""Scenario files: sim/scenarios/<name>.yaml.

A scenario sets the clock window, the bays, starting stock, background sales and a list of
timed events. Times inside a scenario are "HH:MM" on `date` in `tz`.

Events (`do:`), applied at the start of their minute, before that minute's frames:

    sell         bay row position qty      shoppers buy qty packs from the slot (POS records it)
    theft        bay row position qty      packs vanish, no POS line
    restock      bay row position [qty]    staff move packs from the backroom (no qty: fill it)
    face_up      bay row position          staff spread the slot's packs across every facing
    misplace     bay row x_cm sku          a stray pack appears at x_cm
    unmisplace   bay row                   stray packs on the row are taken away
    occlude      bay x [rows] [until] [source]   a person/trolley hides x range [a, b] on rows
                                           (all rows); source: camera = only the camera's view
    quality      bay value [until]         camera frames of the bay get this quality score
    misread      source bay row position units [until]   that source sees `units` packs there
    ambiguous    sku candidates [until]    Vision reports the sku as AMBIGUOUS:<candidates>
    unidentified sku [until]               Vision reports the sku as UNKNOWN (not recognised)
    set_system   sku qty                   the stock system's on-hand count is set to qty

`until` is exclusive and defaults to one minute after `at` (a single frame).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from shelfpulse.contracts import BayId, SkuId

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"

HHMM = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]

REQUIRED = {
    "sell": ("bay", "row", "position", "qty"),
    "theft": ("bay", "row", "position", "qty"),
    "restock": ("bay", "row", "position"),
    "face_up": ("bay", "row", "position"),
    "misplace": ("bay", "row", "x_cm", "sku"),
    "unmisplace": ("bay", "row"),
    "occlude": ("bay", "x"),
    "quality": ("bay", "value"),
    "misread": ("source", "bay", "row", "position", "units"),
    "ambiguous": ("sku", "candidates"),
    "unidentified": ("sku",),
    "set_system": ("sku", "qty"),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SimEvent(_Model):
    at: HHMM
    do: Literal[
        "sell", "theft", "restock", "face_up", "misplace", "unmisplace", "occlude", "quality",
        "misread", "ambiguous", "unidentified", "set_system",
    ]  # fmt: skip
    bay: BayId | None = None
    row: int | None = Field(default=None, ge=0, le=5)
    position: int | None = Field(default=None, ge=0)
    qty: int | None = Field(default=None, ge=0)
    sku: SkuId | None = None
    x_cm: float | None = Field(default=None, ge=0, le=120)
    x: tuple[float, float] | None = None
    rows: list[int] | None = None
    until: HHMM | None = None
    value: float | None = Field(default=None, ge=0, le=1)
    source: Literal["camera", "robot"] | None = None
    units: int | None = Field(default=None, ge=0)
    candidates: list[SkuId] | None = None

    @model_validator(mode="after")
    def _fields(self) -> SimEvent:
        missing = [f for f in REQUIRED[self.do] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.do} at {self.at} needs {missing}")
        if self.x is not None and not 0 <= self.x[0] < self.x[1] <= 120:
            raise ValueError(f"x {self.x} must be [a, b] with 0 <= a < b <= 120")
        if self.candidates is not None and len(self.candidates) < 2:
            raise ValueError("ambiguous needs at least two candidates")
        return self


class StartUnits(_Model):
    bay: BayId
    row: int = Field(ge=0, le=5)
    position: int = Field(ge=0)
    units: int = Field(ge=0)


class Stock(_Model):
    backroom: int = Field(default=0, ge=0)
    system_extra: int = 0  # system on-hand minus real stock (> 0 = phantom stock)


class Scenario(_Model):
    name: str
    description: str = ""
    expect: str = ""  # the outcome the scenario's test checks, in words
    date: str = "2026-10-08"
    tz: str = "+05:30"
    start: HHMM
    end: HHMM
    seed: int = 0
    bays: list[BayId]
    depth_known: bool = True  # camera stereo depth -> depth_left (robot ToF always has it)
    fill: float = Field(default=1.0, ge=0, le=1)  # starting shelf fill, fraction of capacity
    start_units: list[StartUnits] = Field(default_factory=list)  # per-slot override of `fill`
    stock: dict[SkuId, Stock] = Field(default_factory=dict)
    sales_per_hour: dict[SkuId, float] = Field(default_factory=dict)  # random background sales
    events: list[SimEvent] = Field(default_factory=list)

    def at(self, hhmm: str) -> datetime:
        return datetime.fromisoformat(f"{self.date}T{hhmm}:00{self.tz}")

    @property
    def t_start(self) -> datetime:
        return self.at(self.start)

    @property
    def t_end(self) -> datetime:
        return self.at(self.end)

    def until(self, ev: SimEvent) -> datetime:
        return self.at(ev.until) if ev.until else self.at(ev.at) + timedelta(minutes=1)


def load_scenario(name_or_path: str | Path) -> Scenario:
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = SCENARIO_DIR / f"{name_or_path}.yaml"
    sc = Scenario.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    if sc.t_end <= sc.t_start:
        raise ValueError(f"{p}: end must be after start")
    return sc


def scenario_names() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
