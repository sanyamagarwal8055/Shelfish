"""analyze(bay_image) -> BayReading: the single entry point from pixels to the contract.

Step 1 (plumbing): returns a hard-coded but valid reading. Every row is listed and empty, and
quality is 0.0, so the Brain treats the bay as unseen and a stub reading can never raise an alert.
Later steps fill in rectify, detect, identify and depth behind this same signature.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from shelfpulse.contracts import BAY_WIDTH_CM, ROWS, BayReading, Row, Source

STUB_QUALITY = 0.0


def analyze(
    bay_image: np.ndarray,
    bay_id: str,
    source: Source,
    t: datetime,
    frame_ref: str = "",
) -> BayReading:
    """Read one front-on bay image (H x W x 3, BGR) into a BayReading.

    `t` must be timezone-aware. Raises pydantic.ValidationError if the result breaks the contract
    (bad bay_id, naive t, ...).
    """
    if bay_image.ndim < 2 or bay_image.shape[1] == 0:
        raise ValueError(f"bay_image has no width: shape {bay_image.shape}")
    px_per_cm = bay_image.shape[1] / BAY_WIDTH_CM
    return BayReading(
        bay_id=bay_id,
        source=source,
        t=t,
        frame_ref=frame_ref,
        quality=STUB_QUALITY,
        px_per_cm=px_per_cm,
        rows=[Row(row=r) for r in ROWS],
    )
