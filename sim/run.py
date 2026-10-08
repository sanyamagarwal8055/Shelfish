"""Run a scenario minute by minute and write what a perfect Vision track would have sent.

    python -m sim.run --scenario restock_simple --out runs/<id>/ [--brain]
    python -m sim.run --list

Each minute: the minute's scenario events, then background sales (store hours only), then
frames. Camera-tier bays get a camera reading every `sim.camera_interval_s`; every scenario bay
gets a robot reading at each sweep time in configs/robot.yaml. Writes bay_readings.jsonl,
pos.csv and inventory.csv into --out; with --brain, also runs the brain on the readings.

Missions from the Brain do not drive the simulated robot yet (that arrives with the planner).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

from shelfpulse import bus
from shelfpulse.brain import BrainConfig, load_brain_config
from shelfpulse.brain import run as run_brain
from shelfpulse.config import REPO_ROOT, load_robot_config, load_yaml
from shelfpulse.contracts import BayReading, load_sku_master
from shelfpulse.layout import store_map
from shelfpulse.planogram.loader import PlanogramStore
from sim.scenario import Scenario, load_scenario, scenario_names
from sim.store import SimStore

SKU_MASTER = REPO_ROOT / "data" / "sku_master.csv"
MINUTE = timedelta(minutes=1)


def _hours(spec: str) -> tuple[time, time]:
    a, b = (time.fromisoformat(s) for s in spec.split("-"))
    return a, b


def simulate(sc: Scenario, out: Path, cfg: BrainConfig | None = None) -> SimStore:
    cfg = cfg or load_brain_config()
    smap = store_map.load()
    layout = load_yaml("store_layout")
    opens, closes = _hours(layout["store"]["hours"])

    planograms = PlanogramStore(cfg.planograms.dirs)
    plans = {}
    for bay in sc.bays:
        plans[bay] = planograms.get(bay)
        if plans[bay] is None:
            raise ValueError(f"scenario {sc.name}: no planogram for {bay}")
    st = SimStore.build(sc, load_sku_master(SKU_MASTER), plans, float(layout["bay"]["depth_cm"]))

    events = sorted(sc.events, key=lambda e: e.at)  # stable: same-minute events keep file order
    for ev in events:
        if not sc.t_start <= sc.at(ev.at) < sc.t_end:
            raise ValueError(f"scenario {sc.name}: event at {ev.at} is outside the clock window")
    camera_bays = [
        b for b in sc.bays if smap.bays[b].tier == "camera" and smap.bays[b].kind == "bay"
    ]
    sweeps = {datetime.combine(sc.t_start.date(), s, sc.t_start.tzinfo)
              for s in load_robot_config().sweep_times()}  # fmt: skip
    every = timedelta(seconds=cfg.sim.camera_interval_s)

    readings: list[BayReading] = []
    t, i = sc.t_start, 0
    while t < sc.t_end:
        while i < len(events) and sc.at(events[i].at) == t:
            st.apply(events[i], t)
            i += 1
        if opens <= t.timetz().replace(tzinfo=None) < closes:
            st.background_sales(t)
        if (t - sc.t_start) % every == timedelta(0):
            readings += [st.reading(b, "camera", t, cfg.sim.camera_quality, cfg.sim.conf,
                                    cfg.sim.ambiguous_conf) for b in camera_bays]  # fmt: skip
        if t in sweeps:
            readings += [st.reading(b, "robot", t, cfg.sim.robot_quality, cfg.sim.conf,
                                    cfg.sim.ambiguous_conf) for b in sc.bays]  # fmt: skip
        t += MINUTE

    out.mkdir(parents=True, exist_ok=True)
    path = out / "bay_readings.jsonl"
    path.write_text("", encoding="utf-8")
    bus.append_jsonl(path, readings)
    st.write_csvs(out)
    return st


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
    st = simulate(sc, args.out)
    sold = sum(q for _, _, q in st.pos)
    print(f"{sc.name}: wrote {args.out / 'bay_readings.jsonl'}; {sold} packs sold, "
          f"lost sales {st.lost_sales or 0}")  # fmt: skip
    if args.brain:
        cfg = load_brain_config()
        s = run_brain(args.out / "bay_readings.jsonl", args.out, cfg, SKU_MASTER)
        print(f"brain: {s.trusted}/{s.readings} trusted readings, events {dict(s.events)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
