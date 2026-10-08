from datetime import time

from shelfpulse import bus
from shelfpulse.config import load_robot_config
from shelfpulse.contracts import BayReading, parse_bay_reading


def test_robot_config_locked_numbers():
    cfg = load_robot_config()
    assert cfg.sweep_times() == [time(7, 0), time(15, 0)]
    assert cfg.speed_sweep_mps == 0.4
    assert cfg.speed_mission_mps == 0.3
    assert cfg.missions.max_bays == 8
    assert cfg.missions.max_per_hour == 2
    assert cfg.missions.blackout_windows() == [(time(18, 0), time(21, 0))]
    assert cfg.people.max_person_cover == 0.30
    assert cfg.triggers.camera_blocked_min == 15


def test_bus_round_trip(tmp_path, reading):
    path = bus.topic_path("t1", "bay_readings", root=tmp_path)
    assert path == tmp_path / "t1" / "bay_readings.jsonl"
    first = parse_bay_reading(reading)
    bus.append_jsonl(path, first)

    items, offset = bus.tail_jsonl(path, BayReading)
    assert items == [first] and offset == path.stat().st_size

    # A half-written line is not consumed until it is complete.
    second = first.model_copy(update={"bay_id": "G1-L-05"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"partial": ')
    items, offset2 = bus.tail_jsonl(path, BayReading, offset)
    assert items == [] and offset2 == offset
    with open(path, "r+b") as f:  # drop the partial line
        f.truncate(offset)
    bus.append_jsonl(path, [second])
    items, _ = bus.tail_jsonl(path, BayReading, offset)
    assert items == [second]

    assert bus.read_jsonl(path, BayReading) == [first, second]


def test_bus_missing_file(tmp_path):
    assert bus.tail_jsonl(tmp_path / "nope.jsonl", BayReading, 7) == ([], 7)
