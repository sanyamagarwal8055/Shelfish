"""Make the weekly Vision -> Brain fixture: real pipeline output on synthetic shelves.

    python -m tools.demo_fixture --out contracts/fixtures/bay_readings/demo_<date>.jsonl

Runs the current Vision pipeline (detector from configs/perception.yaml, gallery names, depth,
labels) on synthetic shelves drawn from data/gallery, so the Brain can test on what Vision really
emits, mistakes included:
- camera: the three camera-tier demo bays, one frame a minute for three minutes: full shelf,
  then gaps, then gaps + a misplaced pack + a shopper in front (occluded ranges);
- robot: one mission over G1-L-04, G3-L-05, G7-R-02 (robot-only), G9-E-F (end cap) and G6-R-06,
  so labels and robot depth_left appear.
Fixtures must stay under 1 MB, so this is a few dozen readings, not 200.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from shelfpulse.contracts import parse_bay_reading, parse_mission, to_json_dict
from shelfpulse.perception.analyze import load_pipeline
from shelfpulse.perception.run import run as run_camera
from shelfpulse.sources.robot import bridge
from tools.eval_readings import load
from tools.synth.make import make

CAMERA_BAYS = ["G1-L-04", "G1-L-05", "G3-L-05"]
ROBOT_BAYS = ["G1-L-04", "G3-L-05", "G7-R-02", "G9-E-F", "G6-R-06"]
MAX_BYTES = 1_000_000


def build(out: Path, date: str = "2026-10-09", seed: int = 1) -> list[dict]:
    pipeline = load_pipeline()
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stages = [{}, {"gaps": True}, {"gaps": True, "misplaced": True, "occlude": True}]
        for minute, opts in enumerate(stages):
            folder = tmp / f"cam{minute}"
            start = datetime.fromisoformat(f"{date}T10:{minute:02d}:00+05:30")
            make(folder, len(CAMERA_BAYS), seed + minute, depth=True, bays=CAMERA_BAYS,
                 start=start, **opts)  # fmt: skip
            pred = folder / "pred.jsonl"
            run_camera(folder / "images", pred, None, "camera", start=start, pipeline=pipeline,
                       depth_dir=folder / "depth")  # fmt: skip
            lines += [to_json_dict(r) for r in load(pred)]
        mission = parse_mission(
            {"contract_version": "1.0", "mission_id": "M-0100", "kind": "mission",
             "created_at": f"{date}T11:00:00+05:30", "bays": ROBOT_BAYS, "speed_mps": 0.3}
        )  # fmt: skip
        b = bridge.run([mission], tmp / "robot", None, seed=seed, pipeline=pipeline)
        lines += [to_json_dict(r) for r in b.readings]
    lines.sort(key=lambda d: d["t"])
    for d in lines:
        parse_bay_reading(d)
    text = "".join(json.dumps(d) + "\n" for d in lines)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"fixture would be {len(text) // 1000} kB; fixtures must stay < 1 MB")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.demo_fixture", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    lines = build(args.out, args.date, args.seed)
    size = args.out.stat().st_size // 1000
    print(f"wrote {len(lines)} readings ({size} kB) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
