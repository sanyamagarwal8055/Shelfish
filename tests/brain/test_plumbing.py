"""Step 1: brain.py reads BayReadings and writes events, tasks and missions files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from shelfpulse import bus
from shelfpulse.brain import OUTPUT_FILES, load_brain_config, main, run
from shelfpulse.contracts import BayReading, parse_bay_reading, to_json_dict

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "contracts" / "fixtures"
SMOKE = FIXTURES / "scenarios" / "plumbing_smoke.jsonl"
SKUS = ROOT / "data" / "sku_master.csv"
VISION_FIXTURES = sorted((FIXTURES / "bay_readings").glob("*.jsonl"))


def test_config_locked_numbers():
    cfg = load_brain_config()
    assert (cfg.tracker.k, cfg.tracker.n, cfg.tracker.clear_after) == (2, 3, 2)
    assert cfg.fusion.conflict_window_s == 300
    assert cfg.blocked.occluded_frac == 0.5


def test_config_rejects_k_above_n(tmp_path):
    bad = (ROOT / "configs" / "brain.yaml").read_text().replace("k: 2", "k: 4")
    (tmp_path / "b.yaml").write_text(bad)
    with pytest.raises(ValueError):
        load_brain_config(tmp_path / "b.yaml")


def test_smoke_run_writes_outputs(tmp_path):
    out = tmp_path / "run1"
    s = run(SMOKE, out, load_brain_config(), SKUS)
    assert (s.readings, s.trusted, s.untrusted) == (3, 2, 1)
    assert not s.unknown_skus
    for name in OUTPUT_FILES:
        assert (out / name).exists()


def test_outputs_are_overwritten(tmp_path):
    (tmp_path / "events.jsonl").write_text("stale\n")
    run(SMOKE, tmp_path, load_brain_config(), SKUS)
    assert "stale" not in (tmp_path / "events.jsonl").read_text()


def test_unknown_skus_reported(tmp_path, capsys):
    r = bus.read_jsonl(SMOKE, BayReading)[1]
    d = to_json_dict(r)
    d["rows"][0]["packs"][0]["sku"] = "MYSTERY_1KG"
    path = tmp_path / "in.jsonl"
    bus.append_jsonl(path, parse_bay_reading(d))
    assert main(["--readings", str(path), "--out", str(tmp_path / "o")]) == 0
    assert "MYSTERY_1KG" in capsys.readouterr().err


def test_cli(tmp_path, capsys):
    src = tmp_path / "bay_readings.jsonl"
    shutil.copy(SMOKE, src)
    assert main(["--readings", str(src), "--out", str(tmp_path)]) == 0
    assert "3 readings: 2 trusted, 1 below quality threshold" in capsys.readouterr().out


@pytest.mark.parametrize("path", VISION_FIXTURES, ids=lambda p: p.name)
def test_vision_fixture_runs(path, tmp_path):
    s = run(path, tmp_path, load_brain_config(), SKUS)
    assert s.readings > 0
