"""Write the missions fixtures in contracts/fixtures/missions/ from the real robot planner.

    python -m sim.missions_fixture

Gives the Vision robot bridge real planner output to test against:

    demo_missions.jsonl  one routed mission over camera bays, robot-only bays and an end cap,
                         queued for every trigger reason, plus the single-bay missions the brain
                         sends in the occlusion_then_robot and ambiguous scenarios (fast to run)
    demo_sweep.jsonl     the 07:00 sweep alone (bays [] = every bay: slow without recordings)
"""

from __future__ import annotations

import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from shelfpulse import bus
from shelfpulse.brain import load_brain_config
from shelfpulse.config import REPO_ROOT, load_robot_config
from shelfpulse.contracts import Mission
from shelfpulse.layout import store_map
from shelfpulse.robot_planner.mission_queue import MissionQueue
from shelfpulse.robot_planner.scheduler import Scheduler
from sim.run import simulate_with_brain
from sim.scenario import load_scenario

FOLDER = REPO_ROOT / "contracts" / "fixtures" / "missions"
QUEUED = [  # bay, reason
    ("G1-L-05", "blocked"),
    ("G3-L-05", "ambiguous"),
    ("G7-R-02", "verify"),
    ("G6-R-06", "conflict"),
    ("G9-E-F", "promo"),
]


def build() -> list[Mission]:
    cfg, robot = load_brain_config(), load_robot_config()
    sc = load_scenario("restock_simple")  # only for the date and timezone
    s = Scheduler(robot, cfg.planner, cfg.missions.footfall_tx_per_10min_max, store_map.load(),
                  MissionQueue(cfg.planner))  # fmt: skip
    missions = s.tick(sc.at("07:00"))  # sweep
    later = sc.at("09:30")
    for bay, reason in QUEUED:
        s.queue.add(bay, reason, later)
    s.busy_until = None
    missions += s.tick(later + timedelta(minutes=cfg.planner.ready_after_min))
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("occlusion_then_robot", "ambiguous"):
            simulate_with_brain(load_scenario(name), Path(tmp) / name)
            missions += bus.read_jsonl(Path(tmp) / name / "missions.jsonl", Mission)
    # time order, one id sequence across the file
    missions.sort(key=lambda m: m.created_at)
    return [m.model_copy(update={"mission_id": f"M-{i:04d}"}) for i, m in enumerate(missions, 1)]


def main() -> int:
    missions = build()
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, kind in (("demo_missions.jsonl", "mission"), ("demo_sweep.jsonl", "sweep")):
        path = FOLDER / name
        part = [m for m in missions if m.kind == kind]
        path.write_text("", encoding="utf-8")
        bus.append_jsonl(path, part)
        print(f"wrote {len(part)} to {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
