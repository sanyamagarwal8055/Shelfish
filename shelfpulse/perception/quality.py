"""How far to trust one bay image: 0 (don't) to 1. Below 0.5 the Brain treats the bay as unseen.

score = min(sharpness, 1 - occluded share, rows found / 6). Glare is not scored yet.
"""

from __future__ import annotations

import cv2
import numpy as np

from shelfpulse.contracts import BAY_WIDTH_CM, ROWS
from shelfpulse.perception.shelves import Shelf


def sharpness(img: np.ndarray, ref_var: float) -> float:
    """Variance of the Laplacian, scaled so `ref_var` (a sharp image) gives 1."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / ref_var))


def score(img: np.ndarray, shelves: list[Shelf], sharp_ref_var: float = 100.0) -> float:
    if not shelves:
        return 0.0
    hidden = sum(b - a for s in shelves for a, b in s.occluded)
    occluded = hidden / (BAY_WIDTH_CM * len(shelves))
    rows_found = len(shelves) / len(ROWS)
    return round(min(sharpness(img, sharp_ref_var), 1.0 - occluded, rows_found), 3)
