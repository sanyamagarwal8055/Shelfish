"""Every committed fixture under contracts/fixtures/ must obey the contract."""

import json
from pathlib import Path

import pytest

from shelfpulse.contracts import (
    BayReading,
    Mission,
    Planogram,
    RobotStatus,
    check_skus,
    reading_skus,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "contracts" / "fixtures"
DEMO = FIXTURES / "demo_store"
MAX_FIXTURE_BYTES = 1_000_000

# Folder -> models its lines may hold. scenarios/ may mix readings with missions/status.
FOLDER_MODELS = {
    "bay_readings": (BayReading,),
    "missions": (Mission,),
    "scenarios": (BayReading, Mission, RobotStatus),
    "planograms": (Planogram,),
}

JSONL = sorted(FIXTURES.rglob("*.jsonl"))
JSON = sorted(FIXTURES.rglob("*.json"))
ALL = sorted(p for p in FIXTURES.rglob("*") if p.is_file() and p.name != ".gitkeep")


def _rel(p: Path) -> str:
    return p.relative_to(FIXTURES).as_posix()


def _models_for(path: Path) -> tuple:
    for part in reversed(path.relative_to(FIXTURES).parts[:-1]):
        if part in FOLDER_MODELS:
            return FOLDER_MODELS[part]
    raise AssertionError(f"{_rel(path)}: no contract model for this folder")


def _pick(obj: dict, models: tuple):
    if len(models) == 1:
        return models[0]
    if "mission_id" in obj and "kind" in obj:
        return Mission
    if "state" in obj:
        return RobotStatus
    return BayReading


def _check(obj: dict, model, smap, skus, where: str) -> None:
    m = model.model_validate(obj)
    if isinstance(m, BayReading):
        assert m.bay_id in smap.bays, f"{where}: unknown bay {m.bay_id}"
        missing = check_skus(reading_skus(m), skus)
        assert not missing, f"{where}: SKUs not in sku_master: {missing}"
    elif isinstance(m, Planogram):
        assert m.bay_id in smap.bays, f"{where}: unknown bay {m.bay_id}"
        assert Path(where).stem == m.bay_id, f"{where}: file name must be <bay_id>.json"
        missing = check_skus(m.skus(), skus)
        assert not missing, f"{where}: SKUs not in sku_master: {missing}"
    elif isinstance(m, Mission):
        unknown = [b for b in m.bays if b not in smap.bays]
        assert not unknown, f"{where}: unknown bays {unknown}"


@pytest.mark.parametrize("path", JSONL, ids=_rel)
def test_jsonl_fixture(path, smap, skus):
    models = _models_for(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines, start=1):
        if line.strip():
            obj = json.loads(line)
            _check(obj, _pick(obj, models), smap, skus, f"{_rel(path)}:{n}")


@pytest.mark.parametrize("path", JSON, ids=_rel)
def test_json_fixture(path, smap, skus):
    (model,) = _models_for(path)
    _check(json.loads(path.read_text(encoding="utf-8")), model, smap, skus, _rel(path))


@pytest.mark.parametrize("path", ALL, ids=_rel)
def test_fixture_is_small(path):
    assert path.stat().st_size < MAX_FIXTURE_BYTES


def test_demo_store_has_planograms():
    assert len(list((DEMO / "planograms").glob("*.json"))) >= 4


@pytest.mark.parametrize(
    "copy, canonical",
    [("store_layout.yaml", "configs/store_layout.yaml"), ("sku_master.csv", "data/sku_master.csv")],
)
def test_demo_store_copies_match(copy, canonical):
    assert (DEMO / copy).read_bytes() == (ROOT / canonical).read_bytes(), (
        f"contracts/fixtures/demo_store/{copy} drifted from {canonical}; copy it over"
    )
