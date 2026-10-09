"""Reject a bay when people hide too much of it (configs/robot.yaml people.max_person_cover).

Uses the depth-based mask from perception/privacy.py: anything in front of the shelf edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shelfpulse.perception.privacy import people_mask, person_cover
from shelfpulse.perception.shelves import Shelf


@dataclass(frozen=True)
class Verdict:
    cover: float  # share of the bay hidden
    reject: bool
    mask: np.ndarray | None  # pixels to blur before saving


def check(
    depth: np.ndarray | None, shelves: list[Shelf], max_cover: float, margin_mm: float = 50.0
) -> Verdict:
    mask = people_mask(depth, shelves, margin_mm)
    cover = person_cover(mask, shelves)
    return Verdict(round(cover, 3), cover > max_cover, mask)
