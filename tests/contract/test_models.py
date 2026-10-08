import pytest
from pydantic import ValidationError

from shelfpulse.contracts import (
    CONTRACT_VERSION,
    BayReading,
    Mission,
    ambiguous_candidates,
    check_skus,
    is_trusted,
    parse_bay_id,
    parse_bay_reading,
    parse_mission,
    parse_planogram,
    parse_robot_status,
    to_json_dict,
)


def test_version():
    assert CONTRACT_VERSION == "1.0"


def test_valid_reading_round_trips(reading):
    r = parse_bay_reading(reading)
    assert to_json_dict(r) == reading
    assert is_trusted(r)


@pytest.mark.parametrize(
    "path, value",
    [
        (("bay_id",), "G11-L-01"),
        (("bay_id",), "G1-L-00"),
        (("bay_id",), "G1-X-01"),
        (("t",), "2026-10-08T11:20:00"),  # naive time
        (("quality",), 1.2),
        (("source",), "drone"),
        (("contract_version",), "2.0"),
        (("rows", 0, "row"), 6),
        (("rows", 0, "occluded"), [[90.0, 80.0]]),
        (("rows", 0, "packs", 0, "sku"), "rice_1kg"),
        (("rows", 0, "packs", 0, "sku"), "AMBIGUOUS:DAL_1KG"),
        (("rows", 0, "packs", 0, "stack"), 0),
        (("rows", 0, "packs", 0, "x_cm"), 121.0),
        (("rows", 0, "gaps", 0, "w_cm"), 5.0),
        (("rows", 0, "gaps", 0, "x_cm"), 110.0),  # runs past 120 cm
        (("rows", 0, "labels"), [{"x_cm": 34.0, "sku": "RICE_1KG", "price": 89.0}]),  # camera
    ],
)
def test_reading_rejects(reading, path, value):
    target = reading
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        parse_bay_reading(reading)


def test_reading_rejects_unknown_field_and_missing_depth(reading):
    with pytest.raises(ValidationError):
        parse_bay_reading({**reading, "extra": 1})
    del reading["rows"][0]["packs"][0]["depth_left"]
    with pytest.raises(ValidationError):
        parse_bay_reading(reading)


def test_reading_rejects_duplicate_rows(reading):
    reading["rows"].append(dict(reading["rows"][0]))
    with pytest.raises(ValidationError):
        parse_bay_reading(reading)


def test_robot_reading_may_have_labels_and_low_quality(reading):
    reading["source"] = "robot"
    reading["quality"] = 0.3
    reading["rows"][0]["labels"] = [{"x_cm": 34.0, "sku": "RICE_1KG", "price": 89.0}]
    assert not is_trusted(parse_bay_reading(reading))


def test_end_cap_and_ambiguous(reading):
    reading["bay_id"] = "G9-E-F"
    reading["rows"][0]["packs"][0]["sku"] = "AMBIGUOUS:DAL_1KG|DAL_500G"
    reading["rows"][0]["packs"][0]["depth_left"] = None
    assert isinstance(parse_bay_reading(reading), BayReading)


def test_parse_bay_id():
    p = parse_bay_id("G10-R-07")
    assert (p.run, p.face, p.index, p.end, p.is_end_cap) == ("G10", "R", 7, None, False)
    p = parse_bay_id("G2-E-B")
    assert (p.run, p.face, p.index, p.end, p.is_end_cap) == ("G2", "E", None, "B", True)
    with pytest.raises(ValueError):
        parse_bay_id("G1-L-11")


def test_check_skus():
    known = {"DAL_1KG", "RICE_1KG"}
    assert ambiguous_candidates("AMBIGUOUS:DAL_1KG|DAL_500G") == ["DAL_1KG", "DAL_500G"]
    refs = ["RICE_1KG", "UNKNOWN", "AMBIGUOUS:DAL_1KG|DAL_500G"]
    assert check_skus(refs, known) == ["DAL_500G"]


def test_mission(mission):
    assert to_json_dict(parse_mission(mission)) == mission
    sweep = {**mission, "kind": "sweep", "bays": [], "reasons": {}, "speed_mps": 0.4}
    assert isinstance(parse_mission(sweep), Mission)


@pytest.mark.parametrize(
    "change",
    [
        {"bays": [], "reasons": {}},  # a mission needs bays
        {"bays": ["G3-L-05", "G3-L-05"], "reasons": {}},
        {"reasons": {"G1-L-01": "promo"}},  # reason for a bay not in the mission
        {"kind": "patrol"},
        {"speed_mps": 0},
        {"mission_id": "7"},
        {"created_at": "2026-10-08T11:35:00"},
    ],
)
def test_mission_rejects(mission, change):
    with pytest.raises(ValidationError):
        parse_mission({**mission, **change})


def test_robot_status():
    base = {
        "contract_version": "1.0",
        "t": "2026-10-08T11:41:10+05:30",
        "state": "running",
        "x_m": 14.6,
        "y_m": 16.0,
        "mission_id": "M-0007",
        "done_bays": ["G7-R-02"],
        "skipped_bays": [],
        "pending_bays": ["G3-L-05"],
    }
    parse_robot_status(base)
    parse_robot_status({**base, "state": "docked", "mission_id": None, "pending_bays": []})
    with pytest.raises(ValidationError):
        parse_robot_status({**base, "skipped_bays": ["G7-R-02"]})  # done and skipped
    with pytest.raises(ValidationError):
        parse_robot_status({**base, "state": "lost"})


def _slot(pos, sku, a, b, facings=2, min_facings=1):
    return {
        "position": pos,
        "sku_id": sku,
        "x_start_cm": a,
        "x_end_cm": b,
        "facings": facings,
        "min_facings": min_facings,
    }


def test_planogram():
    rows = [[_slot(0, "RICE_1KG", 0, 30), _slot(1, "DAL_1KG", 30, 120)]]
    p = parse_planogram({"bay_id": "G1-L-04", "rows": rows})
    assert p.source == "planogram"
    assert p.skus() == {"RICE_1KG", "DAL_1KG"}


@pytest.mark.parametrize(
    "rows",
    [
        [[_slot(0, "A", 0, 60), _slot(1, "B", 50, 120)]],  # overlap
        [[_slot(0, "A", 60, 120), _slot(1, "B", 0, 60)]],  # out of order
        [[_slot(0, "A", 30, 30)]],  # empty slot
        [[_slot(0, "A", 0, 130)]],  # past bay edge
        [[_slot(0, "A", 0, 60, facings=1, min_facings=2)]],
        [[]] * 7,  # too many rows
    ],
)
def test_planogram_rejects(rows):
    with pytest.raises(ValidationError):
        parse_planogram({"bay_id": "G1-L-04", "rows": rows})
