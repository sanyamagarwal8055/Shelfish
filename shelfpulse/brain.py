"""Program 3: the brain. BayReadings in; events, tasks and missions out.

    python -m shelfpulse.brain --readings runs/<id>/bay_readings.jsonl --out runs/<id>/

Phase 1 reads a whole JSON-lines file in time order and writes events.jsonl, tasks.jsonl and
missions.jsonl into --out (each overwritten). Readings below the contract quality threshold
count as unseen. The reading clock drives everything; there is no datetime.now() here.
"""

from __future__ import annotations

import argparse
import sys
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
from shelfpulse.decision.types import Event, Task

OUTPUT_FILES = ("events.jsonl", "tasks.jsonl", "missions.jsonl")


# --------------------------------------------------------------------------------------------
# Config (configs/brain.yaml)
# --------------------------------------------------------------------------------------------


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrackerCfg(_Cfg):
    k: int = Field(ge=1)
    n: int = Field(ge=1)
    clear_after: int = Field(ge=1)


class FusionCfg(_Cfg):
    conflict_window_s: float = Field(gt=0)


class BlockedCfg(_Cfg):
    occluded_frac: float = Field(gt=0.0, le=1.0)


class MissionsCfg(_Cfg):
    footfall_tx_per_10min_max: int = Field(ge=0)


class BrainConfig(_Cfg):
    tracker: TrackerCfg
    fusion: FusionCfg
    blocked: BlockedCfg
    missions: MissionsCfg


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


@dataclass
class Output:
    events: list[Event] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    missions: list[Mission] = field(default_factory=list)


class Brain:
    def __init__(self, cfg: BrainConfig, known_skus: set[str]):
        self.cfg = cfg
        self.known_skus = known_skus
        self.summary = Summary()

    def handle(self, reading: BayReading) -> Output:
        self.summary.readings += 1
        self.summary.unknown_skus |= set(check_skus(reading_skus(reading), self.known_skus))
        if not is_trusted(reading):
            self.summary.untrusted += 1
            return Output()
        self.summary.trusted += 1
        return Output()  # matcher -> shelf state -> tracker -> decision arrive in later steps


def run(readings_path: Path, out_dir: Path, cfg: BrainConfig, sku_master: Path) -> Summary:
    readings = sorted(bus.read_jsonl(readings_path, BayReading), key=lambda r: r.t)
    brain = Brain(cfg, set(load_sku_master(sku_master)))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in OUTPUT_FILES}
    for p in paths.values():
        p.write_text("", encoding="utf-8")

    for reading in readings:
        out = brain.handle(reading)
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
    if s.unknown_skus:
        print(f"warning: SKUs not in sku_master: {sorted(s.unknown_skus)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
