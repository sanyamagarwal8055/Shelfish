"""Program 3: the brain. BayReadings in; events, tasks and missions out.

    python -m shelfpulse.brain --readings runs/<id>/bay_readings.jsonl --out runs/<id>/

Phase 1 reads a whole JSON-lines file in time order and writes into --out (each overwritten):
observations.jsonl (status per planogram slot), strays.jsonl (misplaced / unknown / ambiguous
packs), events.jsonl, tasks.jsonl and missions.jsonl. Readings below the contract quality
threshold count as unseen. The reading clock drives everything; there is no datetime.now() here.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shelfpulse import bus
from shelfpulse.config import REPO_ROOT, load_yaml
from shelfpulse.contracts import (
    BayReading,
    Mission,
    check_skus,
    is_trusted,
    load_sku_master,
    reading_skus,
)
from shelfpulse.decision.types import Event, SlotObservation, StrayItem, Task
from shelfpulse.planogram.loader import PlanogramStore
from shelfpulse.planogram.matcher import MatchResult, match
from shelfpulse.state.slot_tracker import SlotTracker

OUTPUT_FILES = (
    "observations.jsonl",
    "strays.jsonl",
    "events.jsonl",
    "tasks.jsonl",
    "missions.jsonl",
)


# --------------------------------------------------------------------------------------------
# Config (configs/brain.yaml)
# --------------------------------------------------------------------------------------------


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanogramsCfg(_Cfg):
    dirs: list[str]


class TrackerCfg(_Cfg):
    k: int = Field(ge=1)
    n: int = Field(ge=1)
    clear_after: int = Field(ge=1)
    robot_overrides: bool


class FusionCfg(_Cfg):
    conflict_window_s: float = Field(gt=0)


class BlockedCfg(_Cfg):
    occluded_frac: float = Field(gt=0.0, le=1.0)


class MissionsCfg(_Cfg):
    footfall_tx_per_10min_max: int = Field(ge=0)


class SimCfg(_Cfg):
    camera_interval_s: int = Field(gt=0)
    camera_quality: float = Field(ge=0.0, le=1.0)
    robot_quality: float = Field(ge=0.0, le=1.0)
    conf: float = Field(ge=0.0, le=1.0)
    ambiguous_conf: float = Field(ge=0.0, le=1.0)


class BrainConfig(_Cfg):
    planograms: PlanogramsCfg
    tracker: TrackerCfg
    fusion: FusionCfg
    blocked: BlockedCfg
    missions: MissionsCfg
    sim: SimCfg


def load_brain_config(name_or_path: str | Path = "brain") -> BrainConfig:
    cfg = BrainConfig.model_validate(load_yaml(name_or_path))
    if cfg.tracker.k > cfg.tracker.n:
        raise ValueError("tracker.k cannot exceed tracker.n")
    return cfg


# --------------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------------


@dataclass
class Summary:
    readings: int = 0
    trusted: int = 0
    untrusted: int = 0
    unknown_skus: set[str] = field(default_factory=set)
    no_planogram: set[str] = field(default_factory=set)
    slot_status: Counter[str] = field(default_factory=Counter)
    strays: Counter[str] = field(default_factory=Counter)
    events: Counter[str] = field(default_factory=Counter)


@dataclass
class Output:
    observations: list[SlotObservation] = field(default_factory=list)
    strays: list[StrayItem] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    missions: list[Mission] = field(default_factory=list)


class Brain:
    def __init__(self, cfg: BrainConfig, known_skus: set[str], planograms: PlanogramStore):
        self.cfg = cfg
        self.known_skus = known_skus
        self.planograms = planograms
        self.summary = Summary()
        t = cfg.tracker
        self.tracker = SlotTracker(t.k, t.n, t.clear_after, t.robot_overrides)
        self._meta: dict[Hashable, tuple[int, int | None, str]] = {}  # key -> row, position, sku
        self._live_strays: dict[str, set[Hashable]] = defaultdict(set)  # bay -> watched strays

    def handle(self, reading: BayReading) -> Output:
        self.summary.readings += 1
        self.summary.unknown_skus |= set(check_skus(reading_skus(reading), self.known_skus))
        if not is_trusted(reading):
            self.summary.untrusted += 1
            return Output()
        self.summary.trusted += 1
        plan = self.planograms.get(reading.bay_id)
        if plan is None:
            self.summary.no_planogram.add(reading.bay_id)
            return Output()
        m = match(plan, reading)
        self.summary.slot_status.update(o.status for o in m.slots)
        self.summary.strays.update(s.kind for s in m.strays)
        events = self._track(reading, m)
        self.summary.events.update(e.kind for e in events)
        # shelf state (fusion, quantity) and decision (diagnosis, tasks) arrive in later steps
        return Output(observations=m.slots, strays=m.strays, events=events)

    def _track(self, reading: BayReading, m: MatchResult) -> list[Event]:
        """Feed slot statuses and strays to the k-of-n tracker; return the confirmed changes."""
        reports: list[tuple[Hashable, str | None]] = []
        for o in m.slots:
            if o.status != "UNKNOWN":  # unseen: no vote either way
                key = ("slot", o.bay_id, o.row, o.position)
                self._meta[key] = (o.row, o.position, o.sku)
                reports.append((key, None if o.status == "OK" else o.status))

        present: set[Hashable] = set()
        for s in m.strays:
            if s.kind == "AMBIGUOUS":  # never alerted; the robot planner re-checks these
                continue
            where = s.position if s.position is not None else f"x{int(s.x_cm // 10)}"
            key = ("stray", s.bay_id, s.row, where, s.sku)
            if key not in present:  # several packs of one stray SKU in a slot: one vote
                present.add(key)
                self._meta[key] = (s.row, s.position, s.sku)
                reports.append((key, s.kind))
        live = self._live_strays[reading.bay_id]
        seen_rows = {r.row for r in reading.rows}
        reports += [(key, None) for key in sorted(live - present, key=str) if key[2] in seen_rows]
        live |= present

        events = []
        for key, kind in reports:
            tr = self.tracker.update(key, kind, reading.t, reading.source)
            if key[0] == "stray" and self.tracker.quiet(key):
                live.discard(key)
            if tr is None:
                continue
            row, position, sku = self._meta[key]
            events.append(
                Event(
                    kind=tr.kind,
                    bay_id=reading.bay_id,
                    row=row,
                    position=position,
                    sku=sku,
                    t=tr.t,
                    since=tr.since,
                    previous=tr.previous,
                    source=reading.source,
                )
            )
        return events


def run(readings_path: Path, out_dir: Path, cfg: BrainConfig, sku_master: Path) -> Summary:
    readings = sorted(bus.read_jsonl(readings_path, BayReading), key=lambda r: r.t)
    brain = Brain(cfg, set(load_sku_master(sku_master)), PlanogramStore(cfg.planograms.dirs))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in OUTPUT_FILES}
    for p in paths.values():
        p.write_text("", encoding="utf-8")

    for reading in readings:
        out = brain.handle(reading)
        bus.append_jsonl(paths["observations.jsonl"], out.observations)
        bus.append_jsonl(paths["strays.jsonl"], out.strays)
        bus.append_jsonl(paths["events.jsonl"], out.events)
        bus.append_jsonl(paths["tasks.jsonl"], out.tasks)
        bus.append_jsonl(paths["missions.jsonl"], out.missions)
    return brain.summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shelfpulse.brain", description=__doc__)
    ap.add_argument("--readings", type=Path, required=True, help="bay_readings.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output folder, e.g. runs/<id>/")
    ap.add_argument("--config", default="brain", help="brain config name or path")
    ap.add_argument("--sku-master", type=Path, default=REPO_ROOT / "data" / "sku_master.csv")
    args = ap.parse_args(argv)

    s = run(args.readings, args.out, load_brain_config(args.config), args.sku_master)
    print(f"{s.readings} readings: {s.trusted} trusted, {s.untrusted} below quality threshold")
    print(f"slots: {dict(sorted(s.slot_status.items()))}  strays: {dict(sorted(s.strays.items()))}")
    print(f"events: {dict(sorted(s.events.items()))}")
    if s.no_planogram:
        print(f"warning: no planogram for bays {sorted(s.no_planogram)}", file=sys.stderr)
    if s.unknown_skus:
        print(f"warning: SKUs not in sku_master: {sorted(s.unknown_skus)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
