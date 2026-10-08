"""Program 1, the camera service: shelf-camera frames in, one camera BayReading per bay per minute.

    python -m shelfpulse.sources.camera.service --input runs/seq01/frames --out runs/x/ --speed 60
    python -m shelfpulse.sources.camera.service --input cam.mp4 --camera G1-L-C2 --out runs/<id>/

Input: a folder of time-stamped frames (tools/synth --sequence: frames/<camera>/<n>.jpg, depth maps
alongside, frames.csv with t/camera/files) or one camera's video file (one frame per --every
seconds, no depth). Calibration (calibration.yaml: per camera the lens k1 and each bay's 4 corners
in the undistorted frame) is looked up next to the input unless --calibration is given.

Per frame: undo the lens, warp each bay the camera sees to a front-on 10 px/cm image (and depth),
blur people (anything in front of the shelf edge) before saving it under <out>/frames/<bay>/,
then analyze(source="camera") - quality drops with what people hide - and write the reading to
<out>/bay_readings.jsonl (started fresh; the robot bridge appends to the same file later).
--speed 60 plays 60 simulated minutes per real minute (0 = as fast as possible).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from shelfpulse.contracts import BAY_WIDTH_CM, BayReading, parse_bay_reading, to_json_dict
from shelfpulse.perception.analyze import Pipeline, analyze, default_pipeline
from shelfpulse.perception.privacy import blur_people, people_mask
from shelfpulse.perception.rectify import undistort_radial, warp_to_bay
from shelfpulse.perception.shelves import find_shelves


@dataclass(frozen=True)
class CameraCalib:
    k1: float
    bays: dict[str, list[list[float]]]  # bay_id -> TL, TR, BR, BL in the undistorted frame


@dataclass(frozen=True)
class FrameRef:
    t: datetime
    camera: str
    image: Path
    depth: Path | None


def load_calibration(path: Path) -> dict[str, CameraCalib]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]
    return {c: CameraCalib(float(v.get("k1", 0.0)), v["bays"]) for c, v in data.items()}


def find_calibration(input_path: Path) -> Path:
    base = input_path if input_path.is_dir() else input_path.parent
    for p in (base / "calibration.yaml", base.parent / "calibration.yaml"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"no calibration.yaml next to {input_path}; pass --calibration")


def folder_frames(folder: Path) -> list[FrameRef]:
    """Frames listed in frames.csv (in the folder or its parent), oldest first."""
    for root in (folder, folder.parent):
        index = root / "frames.csv"
        if index.is_file():
            with open(index, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            out = [
                FrameRef(
                    datetime.fromisoformat(r["t"]),
                    r["camera"],
                    root / r["frame_file"],
                    root / r["depth_file"] if r.get("depth_file") else None,
                )
                for r in rows
            ]
            return sorted(out, key=lambda fr: (fr.t, fr.camera))
    raise FileNotFoundError(f"no frames.csv in {folder} or its parent")


def _read(path: Path, flags: int) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


class CameraService:
    def __init__(self, calib: dict[str, CameraCalib], out: Path, pipeline: Pipeline | None = None):
        self.calib = calib
        self.out = out
        self.pipeline = pipeline or default_pipeline()
        self.readings: list[BayReading] = []
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "bay_readings.jsonl"
        self.path.write_text("", encoding="utf-8")  # program 1 starts the run's readings

    def process(
        self, camera: str, t: datetime, image: np.ndarray, depth: np.ndarray | None
    ) -> list[BayReading]:
        if camera not in self.calib:
            raise ValueError(f"no calibration for camera {camera}")
        c = self.calib[camera]
        image = undistort_radial(image, c.k1)
        depth = undistort_radial(depth, c.k1, nearest=True) if depth is not None else None
        out = []
        for bay_id, corners in c.bays.items():
            bay = warp_to_bay(image, corners)
            bdepth = warp_to_bay(depth, corners, nearest=True) if depth is not None else None
            px_per_cm = bay.shape[1] / BAY_WIDTH_CM
            shelves = find_shelves(bay, px_per_cm, self.pipeline.shelves, bdepth)
            mask = people_mask(bdepth, shelves, self.pipeline.shelves.depth_margin_mm)
            safe = blur_people(bay, mask, px_per_cm)  # privacy: blur before saving
            rel = Path("frames") / bay_id / f"{t:%Y-%m-%dT%H-%M-%S}.jpg"
            (self.out / rel.parent).mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", safe)
            if not ok:
                raise ValueError(f"cannot encode {bay_id}")
            buf.tofile(str(self.out / rel))
            reading = analyze(safe, bay_id, "camera", t, rel.as_posix(), self.pipeline, bdepth)
            line = to_json_dict(reading)
            parse_bay_reading(line)  # never write a line the Brain would reject
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            out.append(reading)
        self.readings += out
        return out


def run_folder(
    folder: Path,
    out: Path,
    calibration: Path | None = None,
    speed: float = 0.0,
    pipeline: Pipeline | None = None,
) -> CameraService:
    frames = folder_frames(folder)
    svc = CameraService(load_calibration(calibration or find_calibration(folder)), out, pipeline)
    prev = None
    for fr in frames:
        if speed > 0 and prev is not None and fr.t > prev:
            time.sleep((fr.t - prev).total_seconds() / speed)  # replay at --speed
        prev = fr.t
        depth = _read(fr.depth, cv2.IMREAD_UNCHANGED) if fr.depth and fr.depth.is_file() else None
        svc.process(fr.camera, fr.t, _read(fr.image, cv2.IMREAD_COLOR), depth)
    return svc


def run_video(
    video: Path,
    camera: str,
    out: Path,
    start: datetime,
    every_s: float = 60.0,
    calibration: Path | None = None,
    speed: float = 0.0,
    pipeline: Pipeline | None = None,
) -> CameraService:
    svc = CameraService(load_calibration(calibration or find_calibration(video)), out, pipeline)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(every_s * fps))
        i = 0
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    if speed > 0 and i:
                        time.sleep(every_s / speed)
                    svc.process(camera, start + timedelta(seconds=i / fps), frame, None)
            i += 1
    finally:
        cap.release()
    return svc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shelfpulse.sources.camera.service",
                                 description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    ap.add_argument("--input", type=Path, required=True, help="frames folder or a video file")
    ap.add_argument("--out", type=Path, required=True, help="runs/<id>/")
    ap.add_argument("--speed", type=float, default=0.0, help="replay speed-up (0 = no waiting)")
    ap.add_argument("--calibration", type=Path, help="calibration.yaml (default: next to input)")
    ap.add_argument("--camera", help="video input: which camera it is")
    ap.add_argument("--start", type=datetime.fromisoformat, help="video input: time of frame 0")
    ap.add_argument("--every", type=float, default=60.0, help="video input: seconds between frames")
    args = ap.parse_args(argv)
    try:
        if args.input.is_dir():
            svc = run_folder(args.input, args.out, args.calibration, args.speed)
        else:
            if not args.camera:
                ap.error("--camera is required for a video input")
            start = args.start or datetime.now().astimezone()
            svc = run_video(args.input, args.camera, args.out, start, args.every,
                            args.calibration, args.speed)  # fmt: skip
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {len(svc.readings)} camera readings to {svc.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
