"""Program 3: the brain. BayReadings in; events, tasks and missions out.

    python -m shelfpulse.brain --readings runs/<id>/bay_readings.jsonl --out runs/<id>/

Phase 1 reads a whole JSON-lines file in time order and writes into --out (each overwritten):
observations.jsonl (status per planogram slot), strays.jsonl (misplaced / unknown / ambiguous
packs), events.jsonl, tasks.jsonl and missions.jsonl. Readings below the contract quality
threshold count as unseen. The reading clock drives everything; there is no datetime.now() here.

POS and stock data (pos.csv, inventory.csv) are read from --store-data, by default the folder of
the readings file. Without them alerts still become tasks, with cause NO_STORE_DATA.

The robot planner queues bays for a second look (camera blocked 15+ min, camera/robot
conflict, AMBIGUOUS packs, a sales estimate to verify, a restock on a robot-only bay) and sends
sweeps and missions on the reading clock into missions.jsonl. A robot_status.jsonl next to the
readings, if present, is read too (skipped bays are re-queued).

Shelf labels on robot readings rebuild each bay's label map, written to --label-maps (default
<out>/label_maps/; the store's shared folder is data/label_maps/). A bay with no digital
planogram is matched against its label map; where both exist, persistent differences go to
label_drift.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shelfpulse import bus
from shelfpulse.config import REPO_ROOT, load_robot_config, load_yaml
from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    ROWS,
    BayReading,
    Mission,
    RobotStatus,
    SkuRow,
    check_skus,
    is_trusted,
    load_sku_master,
    reading_skus,
)
from shelfpulse.decision.diagnosis import DiagnosisCfg, diagnose
from shelfpulse.decision.priority import PriorityCfg, bucket, rupees_per_h
from shelfpulse.decision.store_data import StoreData
from shelfpulse.decision.tasks import SLOT_KINDS, Plan, TaskBook, plans_for_slot
from shelfpulse.decision.types import Event, SlotObservation, StrayItem, Task
from shelfpulse.layout import store_map
from shelfpulse.planogram import label_map
from shelfpulse.planogram.label_map import Drift, LabelMapCfg, LabelMaps
from shelfpulse.planogram.loader import PlanogramStore
from shelfpulse.planogram.matcher import MatchResult, match
from shelfpulse.robot_planner.mission_queue import MissionQueue, PlannerCfg
from shelfpulse.robot_planner.scheduler import Scheduler
from shelfpulse.state.quantity import QuantityCfg, low_estimated, shelf_estimate
from shelfpulse.state.shelf_state import ShelfState
from shelfpulse.state.slot_tracker import SlotTracker

OUTPUT_FILES = (
    "observations.jsonl",
    "strays.jsonl",
    "events.jsonl",
    "tasks.jsonl",
    "missions.jsonl",
    "label_drift.jsonl",
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
    diagnosis: DiagnosisCfg
    priority: PriorityCfg
    quantity: QuantityCfg
    planner: PlannerCfg
    label_map: LabelMapCfg
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
    tasks: Counter[str] = field(default_factory=Counter)  # new tasks by action
    recheck: Counter[str] = field(default_factory=Counter)  # robot re-check requests by reason
    recheck_bays: dict[str, str] = field(default_factory=dict)  # at the end: still queued
    missions: Counter[str] = field(default_factory=Counter)  # sent, by kind
    label_maps: set[str] = field(default_factory=set)  # bays whose label map was (re)built
    drift: int = 0
    store_data: bool = False


@dataclass
class Output:
    observations: list[SlotObservation] = field(default_factory=list)
    strays: list[StrayItem] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    missions: list[Mission] = field(default_factory=list)
    drift: list[Drift] = field(default_factory=list)


class Brain:
    def __init__(
        self,
        cfg: BrainConfig,
        skus: dict[str, SkuRow],
        planograms: PlanogramStore,
        store_data: StoreData | None = None,
        shelf_depth_cm: float = 45.0,
        label_map_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.skus = skus
        self.known_skus = set(skus)
        self.planograms = planograms
        self.store_data = store_data
        self.shelf_depth_cm = shelf_depth_cm
        self.summary = Summary(store_data=store_data is not None)
        self.tasks = TaskBook()
        self._seen_tasks: set[str] = set()
        self._last_obs: dict[Hashable, SlotObservation] = {}
        self._history: dict[Hashable, deque[tuple[datetime, int]]] = defaultdict(deque)
        self.shelf = ShelfState(cfg.fusion.conflict_window_s)
        self.smap = store_map.load()
        self.robot = load_robot_config()
        self.queue = MissionQueue(cfg.planner)
        self.scheduler = Scheduler(self.robot, cfg.planner, cfg.missions.footfall_tx_per_10min_max,
                                   self.smap, self.queue, store_data)  # fmt: skip
        self._blocked_since: dict[str, datetime] = {}
        self.label_maps = LabelMaps(skus, cfg.label_map)
        self.label_map_dir = label_map_dir
        t = cfg.tracker
        self.tracker = SlotTracker(t.k, t.n, t.clear_after, t.robot_overrides)
        self._meta: dict[Hashable, tuple[int, int | None, str]] = {}  # key -> row, position, sku
        self._live_strays: dict[str, set[Hashable]] = defaultdict(set)  # bay -> watched strays

    def handle(self, reading: BayReading) -> Output:
        out = self._handle(reading)
        out.missions += self.tick(reading.t)
        return out

    def tick(self, now: datetime) -> list[Mission]:
        """Let the planner send any sweep or mission that is due at `now`."""
        sent = self.scheduler.tick(now)
        self.summary.missions.update(m.kind for m in sent)
        return sent

    def on_robot_status(self, st: RobotStatus) -> None:
        self.scheduler.on_status(st)

    def _handle(self, reading: BayReading) -> Output:
        self.summary.readings += 1
        self.summary.unknown_skus |= set(check_skus(reading_skus(reading), self.known_skus))
        if reading.source == "robot":
            self.queue.visited(reading.bay_id, reading.t)
        else:
            self._watch_blocked(reading)
        if not is_trusted(reading):
            self.summary.untrusted += 1
            return Output()
        self.summary.trusted += 1
        drift = self._read_labels(reading)
        plan = self.planograms.get(reading.bay_id)
        if plan is None:
            self.summary.no_planogram.add(reading.bay_id)
            return Output(drift=drift)
        m = match(plan, reading)
        slots, votes = self._fuse(reading, m.slots)
        self.summary.slot_status.update(o.status for o in slots)
        self.summary.strays.update(s.kind for s in m.strays)
        self._remember(slots)
        events = self._track(reading, m, votes)
        self.summary.events.update(e.kind for e in events)
        tasks = [t for e in events for t in self._decide(e)]
        for t in tasks:
            if t.id not in self._seen_tasks:
                self._seen_tasks.add(t.id)
                self.summary.tasks[t.action] += 1
                self._verify_off_camera(t)
        if reading.source == "camera" and (
            any(s.kind == "AMBIGUOUS" for s in m.strays) or any(o.ambiguous for o in slots)
        ):
            self._queue(reading.bay_id, "ambiguous", reading.t)
        return Output(observations=slots, strays=m.strays, events=events, tasks=tasks,
                      drift=drift)  # fmt: skip

    # --- label maps --------------------------------------------------------------------------

    def _read_labels(self, reading: BayReading) -> list[Drift]:
        """Fold robot shelf labels into the label map; return drift that just became due."""
        known = self.planograms.get(reading.bay_id)
        if known is not None and known.source == "label_map":
            self.label_maps.maps.setdefault(reading.bay_id, known)  # build on the saved map
        changed = self.label_maps.update(reading)
        if changed is not None:
            self.summary.label_maps.add(reading.bay_id)
            self.planograms.put_label_map(changed)
            if self.label_map_dir is not None:
                label_map.write(changed, self.label_map_dir)
        digital = self.planograms.get(reading.bay_id)
        if reading.source != "robot" or digital is None or digital.source == "label_map":
            return []
        drift = self.label_maps.drift(digital, reading.t)
        self.summary.drift += len(drift)
        return drift

    # --- robot planner triggers -------------------------------------------------------------

    def _watch_blocked(self, reading: BayReading) -> None:
        """Camera bay unreadable (quality < 0.5) or mostly hidden for camera_blocked_min ->
        queue it; drop it again once the camera sees the bay."""
        bay = reading.bay_id
        blocked = (
            not is_trusted(reading) or hidden_fraction(reading) > self.cfg.blocked.occluded_frac
        )
        if not blocked:
            self._blocked_since.pop(bay, None)
            self.queue.drop(bay, "blocked")
            return
        since = self._blocked_since.setdefault(bay, reading.t)
        if reading.t - since >= timedelta(minutes=self.robot.triggers.camera_blocked_min):
            self._queue(bay, "blocked", reading.t)

    def _verify_off_camera(self, task: Task) -> None:
        """A restock no camera can see is verified by the robot after verify_after_min."""
        bay = self.smap.bays.get(task.bay)
        if task.action == "RESTOCK" and bay and not (bay.tier == "camera" and bay.kind == "bay"):
            later = task.created_at + timedelta(minutes=self.cfg.planner.verify_after_min)
            self._queue(task.bay, "verify", task.created_at, not_before=later)

    # --- shelf state: camera/robot fusion and the sales-based estimate ----------------------------

    def _fuse(
        self, reading: BayReading, slots: list[SlotObservation]
    ) -> tuple[list[SlotObservation], dict[Hashable, str | None]]:
        """Per slot: UNSURE if camera and robot disagree (no vote, re-check); LOW_ESTIMATED vote
        if it looks fine, depth is unknown and sales say little is left; else its own status."""
        out, votes = [], {}
        for o in slots:
            key = ("slot", o.bay_id, o.row, o.position)
            if o.status == "UNKNOWN":  # unseen: no vote either way
                out.append(o)
                continue
            if self.shelf.conflict(key, reading.source, o.status, reading.t):
                out.append(o.model_copy(update={"status": "UNSURE"}))
                self._queue(o.bay_id, "conflict", reading.t)
                continue
            vote = None if o.status == "OK" else o.status
            if vote is None and o.units is not None and self._counted_low(key, o, reading.t):
                vote = "LOW"  # a measured count that won't last the restock lead time
            elif vote is None and o.units is None and self._estimated_low(o.sku, reading.t):
                # an estimate never downgrades a measured alert
                held = self.tracker.active(key) in ("LOW", "OUT")
                vote = "LOW" if held else "LOW_ESTIMATED"
            votes[key] = vote
            out.append(o)
        return out, votes

    def _velocity(self, sku: str, t: datetime) -> float:
        p = self.cfg.priority
        return self.store_data.velocity(sku, t, p.velocity_window_h, p.min_velocity_span_h)

    def _estimated_low(self, sku: str, t: datetime) -> bool:
        if self.store_data is None:
            return False
        est = shelf_estimate(sku, t, self.store_data)
        return low_estimated(est, self._velocity(sku, t), self.cfg.quantity)

    def _counted_low(self, key: Hashable, o: SlotObservation, t: datetime) -> bool:
        """Measured packs below this slot's share (by planned facings) of the lead-time need."""
        if self.store_data is None or not o.planned_facings:
            return False
        others = sum(x.planned_facings for k, x in self._last_obs.items()
                     if x.sku == o.sku and k != key)  # fmt: skip
        share = o.planned_facings / (others + o.planned_facings)
        q = self.cfg.quantity
        need = (self._velocity(o.sku, t) * q.restock_lead_h + q.safety_units) * share
        return o.units < need

    def _queue(self, bay_id: str, reason: str, t: datetime, not_before: datetime | None = None):
        if self.queue.add(bay_id, reason, t, not_before):
            self.summary.recheck[reason] += 1

    # --- shelf memory for diagnosis --------------------------------------------------------------

    def _remember(self, slots: list[SlotObservation]) -> None:
        """Keep each seen slot's latest observation and its recent pack counts."""
        keep = timedelta(minutes=self.cfg.diagnosis.theft_window_min * 2)
        for o in slots:
            if o.status in ("UNKNOWN", "UNSURE"):
                continue
            key = ("slot", o.bay_id, o.row, o.position)
            self._last_obs[key] = o
            h = self._history[key]
            h.append((o.t, o.units if o.units is not None else o.facings))
            while h and o.t - h[0][0] > keep:
                h.popleft()

    def _shelf_units(self, sku: str) -> int:
        return sum(
            o.units if o.units is not None else o.facings
            for o in self._last_obs.values()
            if o.sku == sku
        )

    # --- decision --------------------------------------------------------------------------------

    def _decide(self, e: Event) -> list[Task]:
        if e.kind == "RESOLVED":
            return self.tasks.verify(e)
        if e.kind == "LOW_ESTIMATED":  # an estimate never makes a staff task: send the robot
            self._queue(e.bay_id, "verify", e.t)
            return []
        sku = self.skus.get(e.sku)
        margin = sku.margin_inr if sku else 0.0
        velocity = 0.0
        if self.store_data is not None:
            p = self.cfg.priority
            velocity = self.store_data.velocity(e.sku, e.t, p.velocity_window_h,
                                                p.min_velocity_span_h)  # fmt: skip
        rupees = rupees_per_h(e.kind, velocity, margin, self.cfg.priority)

        def prio(action: str, r: float) -> str:
            return bucket(r, action, self.cfg.priority)

        if e.kind in SLOT_KINDS:
            key = ("slot", e.bay_id, e.row, e.position)
            obs, history = self._last_obs[key], list(self._history[key])
            depth_cap = max(1, int(self.shelf_depth_cm // sku.depth_cm)) if sku else 1
            now = history[-1][1] if history else 0
            need = max(0, obs.planned_facings * depth_cap - now)
            d = diagnose(e.sku, e.since or e.t, e.t, history, self._shelf_units(e.sku),
                         self.store_data, self.cfg.diagnosis)  # fmt: skip
            plans = plans_for_slot(d, e.bay_id, need, rupees, prio)
        elif e.kind == "MISPLACED":
            home = sku.home_bay if sku else e.bay_id
            plans = [Plan("RETURN", home, 1, "MISPLACED", 0.0, prio("RETURN", 0.0), "staff")]
        else:  # UNKNOWN_ITEM
            plans = [Plan("ENROL", e.bay_id, 1, "UNKNOWN_ITEM", 0.0, prio("ENROL", 0.0), "staff")]
        return self.tasks.upsert(e, plans)

    def _track(
        self, reading: BayReading, m: MatchResult, votes: dict[Hashable, str | None]
    ) -> list[Event]:
        """Feed slot votes and strays to the k-of-n tracker; return the confirmed changes."""
        reports: list[tuple[Hashable, str | None]] = []
        for o in m.slots:
            key = ("slot", o.bay_id, o.row, o.position)
            if key in votes:
                self._meta[key] = (o.row, o.position, o.sku)
                reports.append((key, votes[key]))

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
                    source="estimate" if tr.kind == "LOW_ESTIMATED" else reading.source,
                )
            )
        return events


def hidden_fraction(reading: BayReading) -> float:
    """Share of the bay's 6 rows x 120 cm that is occluded or not in the reading at all."""
    seen = {r.row: r for r in reading.rows}
    hidden = 0.0
    for r in ROWS:
        row = seen.get(r)
        if row is None:
            hidden += BAY_WIDTH_CM
            continue
        cursor = 0.0
        for a, b in sorted(row.occluded):
            hidden += max(0.0, b - max(a, cursor))
            cursor = max(cursor, b)
    return hidden / (BAY_WIDTH_CM * len(ROWS))


def load_store_data(folder: Path | None) -> StoreData | None:
    """pos.csv + inventory.csv from `folder`, or None if either is missing."""
    if folder is None:
        return None
    pos, inv = folder / "pos.csv", folder / "inventory.csv"
    return StoreData.from_csv(pos, inv) if pos.exists() and inv.exists() else None


class Writer:
    """Writes a brain's outputs into a run folder (each file starts empty)."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {name: out_dir / name for name in OUTPUT_FILES}
        for p in self.paths.values():
            p.write_text("", encoding="utf-8")

    def write(self, out: Output) -> None:
        bus.append_jsonl(self.paths["observations.jsonl"], out.observations)
        bus.append_jsonl(self.paths["strays.jsonl"], out.strays)
        bus.append_jsonl(self.paths["events.jsonl"], out.events)
        bus.append_jsonl(self.paths["tasks.jsonl"], out.tasks)
        bus.append_jsonl(self.paths["missions.jsonl"], out.missions)
        bus.append_jsonl(self.paths["label_drift.jsonl"], out.drift)


def make_brain(
    cfg: BrainConfig,
    sku_master: Path,
    store_data: StoreData | None,
    label_map_dir: Path | None = None,
) -> Brain:
    depth = float(load_yaml("store_layout")["bay"]["depth_cm"])
    return Brain(cfg, load_sku_master(sku_master), PlanogramStore(cfg.planograms.dirs),
                 store_data, depth, label_map_dir)  # fmt: skip


def finish(brain: Brain) -> Summary:
    brain.summary.recheck_bays = {bay: e.reason for bay, e in brain.queue.entries.items()}
    return brain.summary


def run(
    readings_path: Path,
    out_dir: Path,
    cfg: BrainConfig,
    sku_master: Path,
    store_data_dir: Path | None | bool = True,
    label_map_dir: Path | None = None,
) -> Summary:
    """store_data_dir: a folder with pos.csv + inventory.csv; True = the readings' folder;
    None/False = no store data. A robot_status.jsonl beside the readings is fed in time order.
    label_map_dir: where label maps are written (default <out_dir>/label_maps)."""
    readings = sorted(bus.read_jsonl(readings_path, BayReading), key=lambda r: r.t)
    status_path = readings_path.parent / "robot_status.jsonl"
    statuses = bus.read_jsonl(status_path, RobotStatus) if status_path.exists() else []
    folder = readings_path.parent if store_data_dir is True else (store_data_dir or None)
    brain = make_brain(cfg, sku_master, load_store_data(folder),
                       label_map_dir or out_dir / "label_maps")  # fmt: skip
    writer = Writer(out_dir)

    feed = sorted([(r.t, 1, r) for r in readings] + [(s.t, 0, s) for s in statuses],
                  key=lambda x: (x[0], x[1]))  # fmt: skip
    for _, _, item in feed:
        if isinstance(item, RobotStatus):
            brain.on_robot_status(item)
        else:
            writer.write(brain.handle(item))
    return finish(brain)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shelfpulse.brain", description=__doc__)
    ap.add_argument("--readings", type=Path, required=True, help="bay_readings.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output folder, e.g. runs/<id>/")
    ap.add_argument("--config", default="brain", help="brain config name or path")
    ap.add_argument("--sku-master", type=Path, default=REPO_ROOT / "data" / "sku_master.csv")
    ap.add_argument("--store-data", type=Path, help="folder with pos.csv + inventory.csv "
                    "(default: the readings' folder)")  # fmt: skip
    ap.add_argument("--no-store-data", action="store_true", help="ignore POS / stock data")
    ap.add_argument("--label-maps", type=Path, help="write label maps here (default "
                    "<out>/label_maps; the store's shared one is data/label_maps)")  # fmt: skip
    args = ap.parse_args(argv)

    store = False if args.no_store_data else (args.store_data or True)
    s = run(args.readings, args.out, load_brain_config(args.config), args.sku_master, store,
            args.label_maps)  # fmt: skip
    print(f"{s.readings} readings: {s.trusted} trusted, {s.untrusted} below quality threshold")
    print(f"slots: {dict(sorted(s.slot_status.items()))}  strays: {dict(sorted(s.strays.items()))}")
    print(f"events: {dict(sorted(s.events.items()))}")
    if s.recheck:
        print(f"robot re-checks queued: {dict(sorted(s.recheck.items()))}")
    if s.missions:
        print(f"robot missions sent: {dict(sorted(s.missions.items()))}")
    if s.label_maps:
        print(f"label maps built from robot labels: {sorted(s.label_maps)}"
              + (f"; {s.drift} drift item(s)" if s.drift else ""))  # fmt: skip
    print(f"new tasks: {dict(sorted(s.tasks.items()))}"
          + ("" if s.store_data else "  (no POS/stock data: causes unknown)"))  # fmt: skip
    if s.no_planogram:
        print(f"warning: no planogram for bays {sorted(s.no_planogram)}", file=sys.stderr)
    if s.unknown_skus:
        print(f"warning: SKUs not in sku_master: {sorted(s.unknown_skus)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
