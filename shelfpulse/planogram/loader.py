"""Find and cache the planogram for a bay.

Folders are searched in order and the first `<bay_id>.json` wins, so a digital planogram
(`data/planograms/`) takes precedence over one rebuilt from robot labels (`data/label_maps/`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from shelfpulse.config import REPO_ROOT
from shelfpulse.contracts import Planogram, parse_planogram


class PlanogramStore:
    def __init__(self, dirs: Iterable[str | Path]):
        self.dirs = [Path(d) if Path(d).is_absolute() else REPO_ROOT / d for d in dirs]
        self._cache: dict[str, Planogram | None] = {}

    def get(self, bay_id: str) -> Planogram | None:
        if bay_id not in self._cache:
            self._cache[bay_id] = self._load(bay_id)
        return self._cache[bay_id]

    def _load(self, bay_id: str) -> Planogram | None:
        for d in self.dirs:
            path = d / f"{bay_id}.json"
            if path.exists():
                plan = parse_planogram(json.loads(path.read_text(encoding="utf-8")))
                if plan.bay_id != bay_id:
                    raise ValueError(f"{path}: holds bay_id {plan.bay_id}")
                return plan
        return None
