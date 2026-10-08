"""Every JSON example in docs/CONTRACT.md must validate against shelfpulse/contracts.py."""

import json
import re
from pathlib import Path

from shelfpulse.contracts import (
    CONTRACT_VERSION,
    parse_bay_reading,
    parse_mission,
    parse_planogram,
    parse_robot_status,
)

DOC = Path(__file__).resolve().parents[2] / "docs" / "CONTRACT.md"
FENCE = "`" * 3

PARSERS = {
    "bay_reading": parse_bay_reading,
    "mission": parse_mission,
    "robot_status": parse_robot_status,
    "planogram": parse_planogram,
}


def _examples() -> list[dict]:
    text = DOC.read_text(encoding="utf-8")
    blocks = re.findall(rf"{FENCE}json\n(.*?){FENCE}", text, flags=re.S)
    return [json.loads(b) for b in blocks]


def _kind(obj: dict) -> str:
    if "mission_id" in obj and "kind" in obj:
        return "mission"
    if "state" in obj:
        return "robot_status"
    if obj.get("source") in ("camera", "robot"):
        return "bay_reading"
    return "planogram"


def test_doc_examples_validate():
    examples = _examples()
    kinds = sorted(_kind(e) for e in examples)
    assert kinds == sorted(PARSERS), f"expected one example per model, got {kinds}"
    for e in examples:
        PARSERS[_kind(e)](e)


def test_doc_version_matches_code():
    first_line = DOC.read_text(encoding="utf-8").splitlines()[0]
    assert f"(v{CONTRACT_VERSION})" in first_line
