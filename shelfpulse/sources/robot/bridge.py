"""Program 2, the robot bridge: Missions in, robot BayReadings and RobotStatus out.

    python -m shelfpulse.sources.robot.bridge --missions runs/<id>/missions.jsonl --out runs/<id>/
        [--recording data/samples/robot] [--person G7-R-02] [--seed 0]

For each Mission (contract section 4) the robot drives the listed bays in the given order ("sweep"
with no bays = every bay of the recording, else every bay in the store) at the mission's speed.
Per bay: frames are stitched into one front-on image + depth (stitcher.py); if people hide more
than configs/robot.yaml people.max_person_cover of it, the bay goes to skipped_bays (the Brain
re-queues it); otherwise it is blurred, saved under <out>/frames/<bay>/ and analysed with
source="robot" (depth gives depth_left). Writes <out>/bay_readings.jsonl and
<out>/robot_status.jsonl (one RobotStatus per change), both appended so they can follow a run's
camera readings. Shelf-edge price labels are read with OCR from the full-resolution stitch
(perception/labels.py) into each row's labels; without the OCR package they stay [].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from shelfpulse.config import load_robot_config
from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    BayReading,
    Mission,
    RobotStatus,
    parse_bay_reading,
    parse_mission,
    parse_robot_status,
    to_json_dict,
)
from shelfpulse.layout import store_map
from shelfpulse.perception.analyze import Pipeline, analyze, default_pipeline
from shelfpulse.perception.labels import LabelReader
from shelfpulse.perception.privacy import blur_people
from shelfpulse.perception.shelves import find_shelves
from shelfpulse.sources.robot import people_guard
from shelfpulse.sources.robot.geometry import Pose
from shelfpulse.sources.robot.stitcher import Stitcher
from shelfpulse.sources.robot.vendor_api import FakeVendorAPI

FRAME_PX_PER_CM = 20.0  # tools/synth/robot.py ROBOT_PX_PER_CM


def read_missions(path: Path) -> list[Mission]:
    with open(path, encoding="utf-8") as f:
        return [parse_mission(json.loads(s)) for s in f if s.strip()]


def _append(path: Path, model) -> None:
    line = to_json_dict(model)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


class Bridge:
    def __init__(
        self,
        api: FakeVendorAPI,
        out: Path,
        pipeline: Pipeline | None = None,
        max_person_cover: float | None = None,
        read_labels: bool = True,
    ):
        self.api = api
        self.out = out
        self.pipeline = pipeline or default_pipeline()
        robot = load_robot_config()
        cover = robot.people.max_person_cover
        self.max_cover = cover if max_person_cover is None else max_person_cover
        self.labels = LabelReader(self.pipeline.skus or {}) if read_labels else None
        self.readings: list[BayReading] = []
        self.statuses: list[RobotStatus] = []

    # --- outputs --------------------------------------------------------------------------

    def _status(self, state: str, mission: Mission | None, done, skipped, pending) -> None:
        t, (x, y) = self.api.status()
        st = RobotStatus(
            t=t,
            state=state,
            x_m=round(x, 3),
            y_m=round(y, 3),
            mission_id=mission.mission_id if mission else None,
            done_bays=list(done),
            skipped_bays=list(skipped),
            pending_bays=list(pending),
        )
        parse_robot_status(to_json_dict(st))
        _append(self.out / "robot_status.jsonl", st)
        self.statuses.append(st)

    def _bays_for(self, m: Mission) -> list[str]:
        if m.bays:
            return list(m.bays)
        rec = self.api.recording
        if rec is not None:
            seen: list[str] = []
            for r in rec.rows:
                x, y, hd = float(r["x_m"]), float(r["y_m"]), float(r["heading_deg"])
                pose = Pose(None, x, y, hd, r["side"])
                for b, centre in self.api.locator.in_view(pose, self.api.frame_width_cm):
                    if b not in seen and 0 <= centre <= BAY_WIDTH_CM:
                        seen.append(b)
            return seen
        return sorted(self.api.smap.bays)

    # --- one mission ----------------------------------------------------------------------

    def run_mission(self, m: Mission) -> RobotStatus:
        bays = self._bays_for(m)
        done: list[str] = []
        skipped: list[str] = []
        self.api.send_waypoints(bays, m.speed_mps, m.created_at)
        self._status("running", m, done, skipped, bays)
        stitcher = Stitcher(self.api.locator, FRAME_PX_PER_CM)
        for ev in self.api.frames():
            if ev.kind == "frame":
                f = ev.frame
                stitcher.add(f.image, f.depth, f.pose)
            elif ev.kind == "passed":
                ok = self._finish_bay(ev.bay_id, stitcher, ev.t)
                (done if ok else skipped).append(ev.bay_id)
                stitcher.drop(ev.bay_id)
                pending = [b for b in bays if b not in done and b not in skipped]
                self._status("running", m, done, skipped, pending)
        self.api.dock()
        self._status("docked", m, done, skipped, [])
        return self.statuses[-1]

    def _finish_bay(self, bay_id: str, stitcher: Stitcher, t) -> bool:
        if not stitcher.complete(bay_id):
            return False  # not fully seen (no frames, or the pass was cut short)
        full, img, depth = stitcher.render(bay_id)
        px_per_cm = img.shape[1] / BAY_WIDTH_CM
        shelves = find_shelves(img, px_per_cm, self.pipeline.shelves, depth)
        verdict = people_guard.check(depth, shelves, self.max_cover,
                                     self.pipeline.shelves.depth_margin_mm)  # fmt: skip
        if verdict.reject:
            return False
        safe = blur_people(img, verdict.mask, px_per_cm)  # privacy: blur before saving
        rel = Path("frames") / bay_id / f"{t:%Y-%m-%dT%H-%M-%S}.jpg"
        (self.out / rel.parent).mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", safe)
        if not ok:
            raise ValueError(f"cannot encode frame for {bay_id}")
        buf.tofile(str(self.out / rel))
        reading = analyze(safe, bay_id, "robot", t, rel.as_posix(), self.pipeline, depth)
        reading = self._with_labels(reading, full, shelves, px_per_cm, stitcher.px_per_cm)
        parse_bay_reading(to_json_dict(reading))  # never write a line the Brain would reject
        _append(self.out / "bay_readings.jsonl", reading)
        self.readings.append(reading)
        return True

    def _with_labels(self, reading: BayReading, full, shelves, px_per_cm, full_px_per_cm):
        if self.labels is None:
            return reading
        try:
            per_row = self.labels.read(full, shelves, px_per_cm, full_px_per_cm)
        except ImportError:
            print("note: rapidocr-onnxruntime not installed; labels stay []", file=sys.stderr)
            self.labels = None
            return reading
        rows = [r.model_copy(update={"labels": per_row.get(r.row, [])}) for r in reading.rows]
        return reading.model_copy(update={"rows": rows})


def run(
    missions: list[Mission],
    out: Path,
    recording: Path | None = None,
    people: set[str] = frozenset(),
    seed: int = 0,
    pipeline: Pipeline | None = None,
    read_labels: bool = True,
) -> Bridge:
    out.mkdir(parents=True, exist_ok=True)
    api = FakeVendorAPI(store_map.load(), recording, people=people, seed=seed)
    bridge = Bridge(api, out, pipeline, read_labels=read_labels)
    for m in sorted(missions, key=lambda m: m.created_at):
        bridge.run_mission(m)
    return bridge


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shelfpulse.sources.robot.bridge",
                                 description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    ap.add_argument("--missions", type=Path, required=True, help="missions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="runs/<id>/")
    ap.add_argument("--recording", type=Path, help="tools/synth/robot.py output (optional)")
    ap.add_argument("--person", default="", help="demo: bays with a shopper in front")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rec = args.recording
    if rec is None and Path("data/samples/robot/pose.csv").is_file():
        rec = Path("data/samples/robot")
    try:
        missions = read_missions(args.missions)
        b = run(missions, args.out, rec, {p for p in args.person.split(",") if p}, args.seed)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    last = b.statuses[-1] if b.statuses else None
    print(
        f"{len(missions)} mission(s): {len(b.readings)} bays read"
        + (f", skipped {last.skipped_bays}" if last and last.skipped_bays else "")
        + f" -> {args.out / 'bay_readings.jsonl'}, {args.out / 'robot_status.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
