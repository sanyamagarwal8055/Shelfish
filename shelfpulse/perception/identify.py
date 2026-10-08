"""Which product is this pack? Gallery search re-ranked by location and size.

For each pack: the top-k SKUs by embedding similarity, each then scored
    score = similarity + location_bonus [planned SKU at this x] + size_bonus * size_match
where the planned SKU comes from data/planograms or data/label_maps (written by the Brain; read
here as a hint, contract section 6) and size_match compares the pack's width in cm with
data/sku_master.csv (height is unreliable: tall packs are cut off by the shelf above).

Output: a real sku_id; "AMBIGUOUS:a|b" when the top two scores are within ambiguous_margin;
"UNKNOWN" when even the best similarity is below unknown_below (location never invents a match).
Pack text (OCR vs pack_keywords) is a planned third hint; `ocr_words` is accepted but unused.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shelfpulse.config import REPO_ROOT
from shelfpulse.contracts import (
    AMBIGUOUS_PREFIX,
    UNKNOWN_SKU,
    Planogram,
    SkuRow,
    parse_planogram,
)
from shelfpulse.perception.embedder import Embedder
from shelfpulse.perception.sku_index import SkuIndex


@dataclass(frozen=True)
class IdentifySettings:
    top_k: int = 5
    unknown_below: float = 0.5
    ambiguous_margin: float = 0.02
    location_bonus: float = 0.05
    size_bonus: float = 0.05
    size_tol: float = 0.25  # relative width error at which size_match falls to ~0.37


class PlanHints:
    """Planned SKU at (bay, row, x) from planogram / label-map JSON files, first folder wins.

    Vision's own small reader of the contract's planogram format (it must not import the Brain's
    loader).
    """

    def __init__(self, dirs: list[str | Path]):
        self.dirs = [Path(d) if Path(d).is_absolute() else REPO_ROOT / d for d in dirs]
        self._cache: dict[str, Planogram | None] = {}

    def get(self, bay_id: str) -> Planogram | None:
        if bay_id not in self._cache:
            self._cache[bay_id] = None
            for d in self.dirs:
                p = d / f"{bay_id}.json"
                if p.is_file():
                    self._cache[bay_id] = parse_planogram(json.loads(p.read_text("utf-8")))
                    break
        return self._cache[bay_id]

    def planned(self, bay_id: str, row: int, x_centre_cm: float) -> str | None:
        plan = self.get(bay_id)
        if plan is None or row >= len(plan.rows):
            return None
        for s in plan.rows[row]:
            if s.x_start_cm <= x_centre_cm <= s.x_end_cm:
                return s.sku_id
        return None


@dataclass
class Identifier:
    embedder: Embedder
    index: SkuIndex
    skus: dict[str, SkuRow]
    hints: PlanHints | None = None
    settings: IdentifySettings = field(default_factory=IdentifySettings)

    def candidates(self, crops: list[np.ndarray]) -> list[list[tuple[str, float]]]:
        if not crops:
            return []
        return self.index.search(self.embedder.embed(crops), self.settings.top_k)

    def decide(
        self,
        cands: list[tuple[str, float]],
        bay_id: str,
        row: int,
        x_cm: float,
        w_cm: float,
        ocr_words: list[str] | None = None,
    ) -> tuple[str, float]:
        """(sku reference, confidence) for one pack from its gallery candidates."""
        s = self.settings
        if not cands or cands[0][1] < s.unknown_below:
            return UNKNOWN_SKU, round(max(cands[0][1], 0.0), 3) if cands else 0.0
        planned = self.hints.planned(bay_id, row, x_cm + w_cm / 2) if self.hints else None
        scored = []
        for sku, sim in cands:
            score = sim
            if sku == planned:
                score += s.location_bonus
            ref = self.skus.get(sku)
            if ref is not None:
                rel = abs(w_cm - ref.width_cm) / ref.width_cm
                score += s.size_bonus * math.exp(-((rel / s.size_tol) ** 2))
            scored.append((score, sim, sku))
        scored.sort(reverse=True)
        (best, best_sim, top), rest = scored[0], scored[1:]
        if rest and best - rest[0][0] < s.ambiguous_margin:
            close = [top] + [k for sc, _, k in rest if best - sc < s.ambiguous_margin]
            return AMBIGUOUS_PREFIX + "|".join(close[:3]), round(min(best_sim, 1.0), 3)
        return top, round(min(max(best_sim, 0.0), 1.0), 3)
