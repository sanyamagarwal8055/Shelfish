import pytest

from shelfpulse.contracts import BAY_ID_RE
from shelfpulse.layout.store_map import DOCK


def test_bay_counts(smap):
    kinds = [b.kind for b in smap.bays.values()]
    assert kinds.count("bay") == 200
    assert kinds.count("end_cap") == 20
    assert len(smap.camera_bays()) == 100
    assert {smap.tier(r) for r in ("G1", "G2", "G3", "G4", "G5")} == {"camera"}
    assert {smap.tier(r) for r in ("G6", "G7", "G8", "G9", "G10")} == {"robot_only"}


def test_bay_ids_match_contract(smap):
    assert all(BAY_ID_RE.match(b) for b in smap.bays)


def test_faces_and_aisles(smap):
    face = smap.bays_of("G1", "L")
    assert [b.bay_id for b in face] == [f"G1-L-{i:02d}" for i in range(1, 11)]
    assert face[0].y0 < face[-1].y0  # bay 01 at the front
    assert smap.bays["G1-L-04"].aisle_id == "A0"
    assert smap.bays["G1-R-04"].aisle_id == "A1"
    assert smap.bays["G2-L-04"].aisle_id == "A1"
    assert smap.bays["G10-R-01"].aisle_id == "A10"
    assert smap.aisle_width_facing("G3-L-05") == pytest.approx(2.0)
    # Aisle centre lines are 2 m wide gaps between runs.
    g1, g2 = smap.runs["G1"], smap.runs["G2"]
    assert g2.x_m - (g1.x_m + g1.depth_m) == pytest.approx(2.0)


def test_home_bays_exist(smap, skus):
    assert len(skus) >= 20
    missing = {s.sku_id: s.home_bay for s in skus.values() if s.home_bay not in smap.bays}
    assert not missing


def test_every_bay_reachable_from_dock(smap):
    for bay_id in smap.bays:
        d = smap.shortest_path_m(DOCK, bay_id)
        assert 0 < d < 150, bay_id


def test_path_uses_walkways(smap):
    # Neighbouring faces across one aisle are a zero-length hop apart.
    assert smap.shortest_path_m("G1-R-04", "G2-L-04") == pytest.approx(0.0)
    # Same run, other face: walk round the end of the run.
    assert smap.shortest_path_m("G1-L-01", "G1-R-01") > 3.0


def test_bay_x_to_store_orientation(smap):
    # Face L is seen looking east: x_cm grows towards the front (-y).
    l0, l1 = smap.bay_x_to_store("G1-L-04", 0), smap.bay_x_to_store("G1-L-04", 120)
    assert l0[1] > l1[1]
    # Face R is seen looking west: x_cm grows towards the back (+y).
    r0, r1 = smap.bay_x_to_store("G1-R-04", 0), smap.bay_x_to_store("G1-R-04", 120)
    assert r0[1] < r1[1]
    assert l1[1] == pytest.approx(r0[1])  # same physical front edge
    with pytest.raises(ValueError):
        smap.bay_x_to_store("G1-L-04", 130)
