"""Run a scenario minute by minute and write what a perfect Vision track would have sent.

    python -m sim.run --scenario restock_simple --out runs/<id>/ [--brain]
    python -m sim.run --list

Each minute: the minute's scenario events, then background sales (store hours only), then
frames. Camera-tier bays get a camera reading every `sim.camera_interval_s`; every scenario bay
gets a robot reading at each sweep time in configs/robot.yaml. Writes bay_readings.jsonl,
pos.csv and inventory.csv into --out.

With --brain the brain runs in lockstep, minute by minute, on live POS and stock, and writes
its outputs into --out too. Its missions drive the simulated robot: it leaves the dock, reaches
each bay after (walking distance / mission speed + bay_scan_s per earlier bay) and reads it,
unless people hide more than robot.yaml people.max_person_cover of it (then the bay is
skipped). One RobotStatus per mission is written to robot_status.jsonl and fed back. Sweep
missions are not re-driven: the simulator already reads every bay at the sweep times.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

from shelfpulse import bus
from shelfpulse.brain import (
    BrainConfig,
    Output,
    Summary,
    Writer,
    finish,
    hidden_fraction,
    load_brain_config,
    make_brain,
)
from shelfpulse.config import REPO_ROOT, load_robot_config, load_yaml
from shelfpulse.contracts import BayReading, Mission, RobotStatus, load_sku_master
from shelfpulse.layout import store_map
from shelfpulse.layout.store_map import DOCK
from shelfpulse.planogram.loader import PlanogramStore
from sim.scenario import Scenario, load_scenario, scenario_names
from sim.store import SimStore

SKU_MASTER = REPO_ROOT / "data" / "sku_master.csv"
MINUTE = timedelta(minutes=1)


def _hours(spec: str) -> tuple[time, time]:
    a, b = (time.fromisoformat(s) for s in spec.split("-"))
    return a, b


def simulate(sc: Scenario, out: Path, cfg: BrainConfig | None = None) -> SimStore:
    """Readings only (no brain)."""
    return _simulate(sc, out, cfg or load_brain_config(), with_brain=False)[0]


def simulate_with_brain(
    sc: Scenario, out: Path, cfg: BrainConfig | None = None
) -> tuple[SimStore, Summary]:
    """Readings, with the brain in the loop; its missions drive the simulated robot."""
    return _simulate(sc, out, cfg or load_brain_config(), with_brain=True)


def _simulate(sc: Scenario, out: Path, cfg: BrainConfig, with_brain: bool):
    smap = store_map.load()
    robot = load_robot_config()
    layout = load_yaml("store_layout")
    opens, closes = _hours(layout["store"]["hours"])

    planograms = PlanogramStore(cfg.planograms.dirs)
    plans = {}
    for bay in sc.bays:
        plans[bay] = planograms.get(bay)
        if plans[bay] is None:
            raise ValueError(f"scenario {sc.name}: no planogram for {bay}")
    st = SimStore.build(sc, load_sku_master(SKU_MASTER), plans, float(layout["bay"]["depth_cm"]))
    st.dock_xy = smap.dock_xy

    events = sorted(sc.events, key=lambda e: e.at)  # stable: same-minute events keep file order
    for ev in events:
        if not sc.t_start <= sc.at(ev.at) < sc.t_end:
            raise ValueError(f"scenario {sc.name}: event at {ev.at} is outside the clock window")
    camera_bays = [
        b for b in sc.bays if smap.bays[b].tier == "camera" and smap.bays[b].kind == "bay"
    ]
    sweeps = {datetime.combine(sc.t_start.date(), s, sc.t_start.tzinfo)
              for s in robot.sweep_times()}  # fmt: skip
    every = timedelta(seconds=cfg.sim.camera_interval_s)

    def read(bay: str, source: str, t: datetime) -> BayReading:
        q = cfg.sim.camera_quality if source == "camera" else cfg.sim.robot_quality
        return st.reading(bay, source, t, q, cfg.sim.conf, cfg.sim.ambiguous_conf)

    out.mkdir(parents=True, exist_ok=True)
    brain = make_brain(cfg, SKU_MASTER, st.data, out / "label_maps") if with_brain else None
    writer = Writer(out) if with_brain else None
    readings: list[BayReading] = []
    statuses: list[RobotStatus] = []
    pending: list[tuple[datetime, int, object]] = []  # robot visits and statuses still to happen

    def feed(item) -> None:
        if isinstance(item, RobotStatus):
            statuses.append(item)
            if brain:
                brain.on_robot_status(item)
            return
        readings.append(item)
        if brain:
            result = brain.handle(item)
            writer.write(result)
            for m in result.missions:
                if m.kind == "mission":
                    pending.extend(_drive(m, smap, cfg.planner.bay_scan_s))

    t, i = sc.t_start, 0
    while t < sc.t_end:
        while i < len(events) and sc.at(events[i].at) == t:
            st.apply(events[i], t)
            i += 1
        if opens <= t.timetz().replace(tzinfo=None) < closes:
            st.background_sales(t)
        if (t - sc.t_start) % every == timedelta(0):
            for b in camera_bays:
                feed(read(b, "camera", t))
        if t in sweeps:
            for b in sc.bays:
                feed(read(b, "robot", t))
        while True:  # robot visits during this minute (a new mission may add more)
            due = sorted((p for p in pending if p[0] < t + MINUTE), key=lambda p: (p[0], p[1]))
            if not due:
                break
            item = due[0]
            pending.remove(item)
            result = _visit(item, st, read, robot.people.max_person_cover)
            if result is not None:
                feed(result)
        if brain:  # the planner also ticks on minutes with no frames (robot-only bays)
            for m in brain.tick(t):
                writer.write(Output(missions=[m]))
                if m.kind == "mission":
                    pending.extend(_drive(m, smap, cfg.planner.bay_scan_s))
        t += MINUTE

    path = out / "bay_readings.jsonl"
    path.write_text("", encoding="utf-8")
    bus.append_jsonl(path, readings)
    if statuses:
        spath = out / "robot_status.jsonl"
        spath.write_text("", encoding="utf-8")
        bus.append_jsonl(spath, statuses)
    st.write_csvs(out)
    return st, (finish(brain) if brain else None)


def _drive(m: Mission, smap, bay_scan_s: float) -> list[tuple[datetime, int, object]]:
    """When the robot reaches each bay of a mission, and when it is back at the dock."""
    plan, here, clock = [], DOCK, 0.0
    for k, bay in enumerate(m.bays):
        clock += smap.shortest_path_m(here, bay) / m.speed_mps + (bay_scan_s if k else 0.0)
        plan.append((m.created_at + timedelta(seconds=clock), 1, ("visit", m, bay)))
        here = bay
    clock += smap.shortest_path_m(here, DOCK) / m.speed_mps + bay_scan_s
    plan.append((m.created_at + timedelta(seconds=clock), 2, ("docked", m, None)))
    return plan


def _visit(item, st: SimStore, read, max_cover: float) -> BayReading | RobotStatus | None:
    t, _, (what, m, bay) = item
    log = st.robot_log.setdefault(m.mission_id, {"done": [], "skipped": []})
    if what == "docked":
        return RobotStatus(t=t, state="docked", x_m=st.dock_xy[0], y_m=st.dock_xy[1],
                           mission_id=m.mission_id, done_bays=log["done"],
                           skipped_bays=log["skipped"], pending_bays=[])  # fmt: skip
    reading = read(bay, "robot", t)
    if hidden_fraction(reading) > max_cover:  # people in the way: skipped, no reading sent
        log["skipped"].append(bay)
        return None
    log["done"].append(bay)
    return reading


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.run", description=__doc__)
    ap.add_argument("--scenario", help="name in sim/scenarios/ or a .yaml path")
    ap.add_argument("--out", type=Path, help="output folder, e.g. runs/<id>/")
    ap.add_argument("--brain", action="store_true", help="also run the brain on the readings")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name in scenario_names():
            print(f"{name:24} {load_scenario(name).description}")
        return 0
    if not args.scenario or not args.out:
        ap.error("--scenario and --out are required")

    sc = load_scenario(args.scenario)
    if args.brain:
        st, s = simulate_with_brain(sc, args.out)
    else:
        st, s = simulate(sc, args.out), None
    sold = sum(q for _, _, q in st.pos)
    print(f"{sc.name}: wrote {args.out / 'bay_readings.jsonl'}; {sold} packs sold, "
          f"lost sales {st.lost_sales or 0}")  # fmt: skip
    if s:
        print(f"brain: {s.trusted}/{s.readings} trusted readings, events {dict(s.events)}, "
              f"tasks {dict(s.tasks)}, missions {dict(s.missions)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
