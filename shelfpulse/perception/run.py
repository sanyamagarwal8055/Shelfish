"""CLI: a folder of bay images or a video in, one BayReading JSON line per image out.

    python -m shelfpulse.perception.run --input <folder|video> --out runs/<id>/bay_readings.jsonl

Folder: every .jpg/.jpeg/.png, sorted by name. t = --start + 1 min per image, else the file's
modified time. Video: one frame every --every seconds; t = --start (else the file's modified
time) + offset into the video.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from shelfpulse.config import load_yaml
from shelfpulse.contracts import BAY_ID_RE, parse_bay_reading, to_json_dict
from shelfpulse.perception.analyze import Pipeline, analyze, load_pipeline
from shelfpulse.perception.rectify import undistort_radial

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
FOLDER_STEP = timedelta(minutes=1)
DEFAULT_BAY = "G1-L-04"
_BAY_IN_NAME = re.compile(BAY_ID_RE.pattern.strip("^$"))


def bay_from_name(frame_ref: str) -> str | None:
    """`images/0003_G1-L-04.jpg` -> "G1-L-04" (tools/synth names frames this way)."""
    m = _BAY_IN_NAME.search(Path(frame_ref.split("#", 1)[0]).stem)
    return m.group(0) if m else None


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def _read_image(path: Path) -> np.ndarray:
    # np.fromfile + imdecode, because cv2.imread fails on non-ASCII Windows paths.
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def iter_folder(folder: Path, start: datetime | None) -> Iterator[tuple[np.ndarray, datetime, str]]:
    # Not a generator itself, so an empty folder fails before the output file is created.
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise ValueError(f"no {sorted(IMAGE_EXTS)} images in {folder}")

    def frames() -> Iterator[tuple[np.ndarray, datetime, str]]:
        for i, p in enumerate(paths):
            t = start + i * FOLDER_STEP if start else _mtime(p)
            yield _read_image(p), t, p.as_posix()

    return frames()


def iter_video(
    video: Path, start: datetime | None, every_s: float
) -> Iterator[tuple[np.ndarray, datetime, str]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(every_s * fps))
        t0 = start or _mtime(video)
        i = 0
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    sec = i / fps
                    yield frame, t0 + timedelta(seconds=sec), f"{video.as_posix()}#t={sec:.1f}"
            i += 1
    finally:
        cap.release()


def run(
    input_path: Path,
    out: Path,
    bay_id: str | None,
    source: str,
    start: datetime | None = None,
    every_s: float = 60.0,
    undistort_k1: float = 0.0,
    pipeline: Pipeline | None = None,
) -> int:
    """Analyse every image/frame, write validated JSON lines to `out`. Returns the line count.

    bay_id None: taken from each file name if it contains one, else G1-L-04.
    """
    if input_path.is_dir():
        frames = iter_folder(input_path, start)
    elif input_path.suffix.lower() in VIDEO_EXTS:
        frames = iter_video(input_path, start, every_s)
    else:
        raise ValueError(f"--input must be a folder of images or a video, got {input_path}")

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for img, t, ref in frames:
            img = undistort_radial(img, undistort_k1)
            bay = bay_id or bay_from_name(ref) or DEFAULT_BAY
            reading = analyze(img, bay, source, t, frame_ref=ref, pipeline=pipeline)
            line = to_json_dict(reading)
            parse_bay_reading(line)  # never write a line the Brain would reject
            f.write(json.dumps(line) + "\n")
            n += 1
    if n == 0:
        raise ValueError(f"no frames read from {input_path}")
    return n


def _aware(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--start needs a timezone, e.g. {s}+05:30")
    return t


def _bay_id(s: str) -> str:
    if not BAY_ID_RE.match(s):
        raise argparse.ArgumentTypeError(f"invalid bay_id {s!r} (expected e.g. G1-L-04)")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m shelfpulse.perception.run", description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="folder of images or a video")
    ap.add_argument("--out", required=True, type=Path, help="runs/<id>/bay_readings.jsonl")
    ap.add_argument("--bay-id", type=_bay_id, help="default: from the file name, else G1-L-04")
    ap.add_argument("--source", default="camera", choices=["camera", "robot"])
    ap.add_argument("--start", type=_aware, help="ISO time of the first frame, with timezone")
    ap.add_argument("--every", type=float, default=60.0, help="video: seconds between frames")
    ap.add_argument("--detector", choices=["classic", "yolo"], help="override the config backend")
    ap.add_argument(
        "--undistort-synth",
        action="store_true",
        help="undo tools/synth --distort (rectify.synth_k1 in configs/perception.yaml) first",
    )
    args = ap.parse_args(argv)
    k1 = load_yaml("perception")["rectify"]["synth_k1"] if args.undistort_synth else 0.0
    try:
        pipeline = load_pipeline(backend=args.detector)
        n = run(
            args.input, args.out, args.bay_id, args.source, args.start, args.every, k1, pipeline
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {n} readings to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
