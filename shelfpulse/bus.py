"""Message passing between programs (shared). Phase 1: JSON-lines files under runs/<run_id>/.

Each topic is one append-only file of contract objects, one per line. A reader either loads the
whole file (`read_jsonl`) or polls it for new complete lines (`tail_jsonl`). Phase 2 swaps this
for streams behind the same topic names.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from shelfpulse.contracts import BayReading, Mission, RobotStatus, to_json_dict

M = TypeVar("M", bound=BaseModel)

RUNS_DIR = Path("runs")

TOPIC_FILES = {
    "bay_readings": "bay_readings.jsonl",
    "missions": "missions.jsonl",
    "robot_status": "robot_status.jsonl",
}
TOPIC_MODELS: dict[str, type[BaseModel]] = {
    "bay_readings": BayReading,
    "missions": Mission,
    "robot_status": RobotStatus,
}


def run_dir(run_id: str, root: str | Path = RUNS_DIR) -> Path:
    """runs/<run_id>/, created if missing."""
    d = Path(root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def topic_path(run_id: str, topic: str, root: str | Path = RUNS_DIR) -> Path:
    return run_dir(run_id, root) / TOPIC_FILES[topic]


def append_jsonl(path: str | Path, items: BaseModel | Iterable[BaseModel]) -> None:
    """Append one model or several, one compact JSON object per line."""
    if isinstance(items, BaseModel):
        items = [items]
    lines = "".join(json.dumps(to_json_dict(m), ensure_ascii=False) + "\n" for m in items)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(lines)


def read_jsonl(path: str | Path, model: type[M]) -> list[M]:
    """Load and validate every line. Errors name the file and line number."""
    out: list[M] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                out.append(_parse(line, model, path, n))
    return out


def tail_jsonl(path: str | Path, model: type[M], offset: int = 0) -> tuple[list[M], int]:
    """Read complete lines written since byte `offset`. Returns (items, new_offset).

    A partly written last line is left for the next call. A missing file yields ([], offset).
    """
    p = Path(path)
    if not p.exists():
        return [], offset
    with open(p, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1  # 0 if no complete line yet
    out = [
        _parse(line.decode("utf-8"), model, path, None)
        for line in data[:end].splitlines()
        if line.strip()
    ]
    return out, offset + end


def _parse(line: str, model: type[M], path: str | Path, lineno: int | None) -> M:
    try:
        return model.model_validate(json.loads(line))
    except Exception as e:
        where = f"{path}:{lineno}" if lineno else str(path)
        raise ValueError(f"{where}: invalid {model.__name__}: {e}") from e
