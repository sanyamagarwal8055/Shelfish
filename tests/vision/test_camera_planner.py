"""Step 3: camera count and positions from the store layout."""

from __future__ import annotations

import pytest
import yaml

from scripts.plan_cameras import main
from shelfpulse.config import load_yaml
from shelfpulse.layout import store_map
from shelfpulse.layout.camera_planner import (
    load_settings,
    plan,
    positions,
    total_cameras,
    useful_width,
)


@pytest.fixture(scope="module")
def smap():
    return store_map.load()


def test_useful_width_2m_aisle_is_angle_limited():
    assert useful_width(2.0) == pytest.approx(2.8, abs=0.001)  # 2 * 2.0 * tan(35 deg)


def test_2m_aisle_12m_run_gives_5_cameras():
    assert positions(12.0, 2.0) == pytest.approx([1.2, 3.6, 6.0, 8.4, 10.8])


def test_1_5m_aisle_gives_7_cameras():
    assert len(positions(12.0, 1.5)) == 7


def test_settings_come_from_config():
    s = load_settings()
    assert (s.hfov_deg, s.sensor_px, s.max_angle_deg, s.overlap_m) == (90, 4000, 35, 0.2)


def test_store_total_is_50_and_every_camera_bay_seen_whole(smap):
    plans = plan(smap, s=load_settings())
    assert total_cameras(plans) == 50
    assert {p.run for p in plans} == {"G1", "G2", "G3", "G4", "G5"}
    seen = {b for p in plans for c in p.cameras for b in c.bays}
    assert seen == set(smap.camera_bays())
    assert all(not p.unseen_bays for p in plans)


def test_camera_details(smap):
    g1l, g1r = plan(smap, ["G1"])
    c1 = g1l.cameras[0]
    assert c1.id == "G1-L-C1" and c1.aisle_id == "A0" and c1.mount == "pole"
    assert c1.bays == ("G1-L-01", "G1-L-02")
    assert g1r.cameras[0].mount == "G2"
    run = smap.runs["G1"]
    assert c1.xy_m == pytest.approx((run.x_m - 2.0, run.y_m + 1.2))


def test_narrow_aisle_warns_about_split_bays(tmp_path):
    layout = load_yaml("store_layout")
    layout["aisles"]["width_m"] = 1.5
    path = tmp_path / "narrow.yaml"
    path.write_text(yaml.safe_dump(layout), encoding="utf-8")
    plans = plan(store_map.load(path), ["G1"])
    assert [len(p.cameras) for p in plans] == [7, 7]
    assert all(p.unseen_bays for p in plans)


def test_script_prints_total_and_writes_yaml(tmp_path, capsys):
    out = tmp_path / "cameras.yaml"
    assert main(["--out", str(out)]) == 0
    assert "TOTAL 50 cameras" in capsys.readouterr().out
    cams = yaml.safe_load(out.read_text(encoding="utf-8"))["cameras"]
    assert len(cams) == 50 and len({c["id"] for c in cams}) == 50
