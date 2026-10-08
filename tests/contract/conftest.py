import copy
from pathlib import Path

import pytest

from shelfpulse.contracts import load_sku_master
from shelfpulse.layout import store_map

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "contracts" / "fixtures"

BAY_READING = {
    "contract_version": "1.0",
    "bay_id": "G1-L-04",
    "source": "camera",
    "t": "2026-10-08T11:20:00+05:30",
    "frame_ref": "frames/G1-L-04/2026-10-08T11-20-00.jpg",
    "quality": 0.92,
    "px_per_cm": 10.0,
    "rows": [
        {
            "row": 3,
            "occluded": [[80.0, 120.0]],
            "packs": [
                {
                    "sku": "RICE_1KG",
                    "conf": 0.91,
                    "x_cm": 34.0,
                    "w_cm": 9.5,
                    "h_cm": 26.0,
                    "stack": 1,
                    "depth_left": 3,
                },
            ],
            "gaps": [{"x_cm": 43.5, "w_cm": 17.5}],
            "labels": [],
        }
    ],
}

MISSION = {
    "contract_version": "1.0",
    "mission_id": "M-0007",
    "kind": "mission",
    "created_at": "2026-10-08T11:35:00+05:30",
    "bays": ["G3-L-05", "G7-R-02"],
    "reasons": {"G3-L-05": "blocked", "G7-R-02": "verify"},
    "speed_mps": 0.3,
}


@pytest.fixture
def reading() -> dict:
    return copy.deepcopy(BAY_READING)


@pytest.fixture
def mission() -> dict:
    return copy.deepcopy(MISSION)


@pytest.fixture(scope="session")
def smap():
    return store_map.load()


@pytest.fixture(scope="session")
def skus():
    return load_sku_master(ROOT / "data" / "sku_master.csv")
