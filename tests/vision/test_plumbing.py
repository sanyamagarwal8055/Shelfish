"""Step 1: analyze() and the run CLI produce BayReading lines that validate against the contract."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from shelfpulse.contracts import ROWS, parse_bay_reading
from shelfpulse.perception.analyze import analyze
from shelfpulse.perception.run import main

IST = timezone(timedelta(hours=5, minutes=30))
T = datetime(2026, 10, 8, 11, 20, tzinfo=IST)


def _bay_image(w: int = 1200, h: int = 2100) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


@pytest.fixture
def three_images(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(3):
        cv2.imwrite(str(folder / f"bay_{i}.jpg"), _bay_image(240, 420))
    (folder / "notes.txt").write_text("not an image")
    return folder


def _lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("source", ["camera", "robot"])
def test_analyze_is_valid(source):
    r = analyze(_bay_image(), bay_id="G1-L-04", source=source, t=T, frame_ref="x.jpg")
    parse_bay_reading(r.model_dump(mode="json"))
    assert r.px_per_cm == 10.0
    assert [row.row for row in r.rows] == list(ROWS)
    assert all(row.labels == [] for row in r.rows)


def test_stub_is_untrusted():
    # Quality 0 means the Brain treats the bay as unseen, so stub output can't raise alerts.
    r = analyze(_bay_image(), bay_id="G1-L-04", source="camera", t=T)
    assert r.quality < 0.5


def test_analyze_rejects_bad_bay_id():
    with pytest.raises(ValidationError):
        analyze(_bay_image(), bay_id="G11-L-04", source="camera", t=T)


def test_cli_folder_writes_one_valid_line_per_image(three_images, tmp_path):
    out = tmp_path / "runs" / "t1" / "bay_readings.jsonl"
    assert main(["--input", str(three_images), "--out", str(out), "--start", T.isoformat()]) == 0
    lines = _lines(out)
    assert len(lines) == 3
    readings = [parse_bay_reading(d) for d in lines]
    assert [r.t for r in readings] == [T, T + timedelta(minutes=1), T + timedelta(minutes=2)]
    assert [r.frame_ref.rsplit("/", 1)[-1] for r in readings] == [f"bay_{i}.jpg" for i in range(3)]
    assert all(r.px_per_cm == 2.0 for r in readings)


def test_cli_uses_file_time_without_start(three_images, tmp_path):
    out = tmp_path / "out.jsonl"
    assert main(["--input", str(three_images), "--out", str(out), "--source", "robot"]) == 0
    readings = [parse_bay_reading(d) for d in _lines(out)]
    assert all(r.t.tzinfo is not None and r.source == "robot" for r in readings)


def test_cli_video(tmp_path):
    video = tmp_path / "walk.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (240, 420))
    for _ in range(25):  # 2.5 s at 10 fps
        vw.write(_bay_image(240, 420))
    vw.release()
    out = tmp_path / "out.jsonl"
    args = ["--input", str(video), "--out", str(out), "--start", T.isoformat(), "--every", "1"]
    assert main(args) == 0
    readings = [parse_bay_reading(d) for d in _lines(out)]
    assert [r.t - T for r in readings] == [timedelta(seconds=s) for s in (0, 1, 2)]


def test_cli_empty_folder_fails(tmp_path):
    out = tmp_path / "out.jsonl"
    assert main(["--input", str(tmp_path), "--out", str(out)]) == 1
    assert not out.exists()


def test_cli_rejects_bad_bay_id(three_images, tmp_path):
    with pytest.raises(SystemExit):
        main(["--input", str(three_images), "--out", str(tmp_path / "o.jsonl"), "--bay-id", "X"])
