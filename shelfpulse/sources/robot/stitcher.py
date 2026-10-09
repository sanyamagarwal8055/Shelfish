"""Robot frames + poses -> one front-on image (and depth map) per bay.

Frames are front-on strips (the mast camera faces the shelf square on) and the pose says where
the robot was, so a frame's place in a bay is known: no feature matching. Each frame is pasted at
its x_cm in every bay it overlaps; a bay is complete once its full 0-120 cm width is covered.
The mosaic keeps the frames' resolution (for reading labels); render() also gives a 10 px/cm copy
for analyze().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from shelfpulse.contracts import BAY_WIDTH_CM
from shelfpulse.sources.robot.geometry import BayLocator, Pose


@dataclass
class _Mosaic:
    image: np.ndarray
    depth: np.ndarray
    covered: np.ndarray  # bool per column
    last_t: object = None  # time of the latest frame


@dataclass
class Stitcher:
    locator: BayLocator
    px_per_cm: float  # frame resolution
    out_px_per_cm: float = 10.0
    _bays: dict[str, _Mosaic] = field(default_factory=dict)

    def add(self, image: np.ndarray, depth: np.ndarray, pose: Pose) -> list[str]:
        """Paste a frame into every bay it shows; returns those bay_ids."""
        h, fw = image.shape[:2]
        width_cm = fw / self.px_per_cm
        bay_w = round(BAY_WIDTH_CM * self.px_per_cm)
        seen = []
        for bay_id, centre_cm in self.locator.in_view(pose, width_cm):
            m = self._bays.get(bay_id)
            if m is None:
                m = _Mosaic(
                    np.zeros((h, bay_w, 3), np.uint8),
                    np.zeros((h, bay_w), np.uint16),
                    np.zeros(bay_w, bool),
                )
                self._bays[bay_id] = m
            left = round((centre_cm - width_cm / 2) * self.px_per_cm)  # bay column of frame col 0
            a, b = max(left, 0), min(left + fw, bay_w)
            if a >= b:
                continue
            m.image[:, a:b] = image[:h, a - left : b - left]
            m.depth[:, a:b] = depth[:h, a - left : b - left]
            m.covered[a:b] = True
            m.last_t = pose.t
            seen.append(bay_id)
        return seen

    def coverage(self, bay_id: str) -> float:
        m = self._bays.get(bay_id)
        return float(m.covered.mean()) if m is not None else 0.0

    def complete(self, bay_id: str) -> bool:
        return self.coverage(bay_id) >= 1.0

    def last_time(self, bay_id: str):
        m = self._bays.get(bay_id)
        return m.last_t if m else None

    def render(self, bay_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(full-res image, 10 px/cm image, 10 px/cm depth) of a bay."""
        m = self._bays[bay_id]
        k = self.out_px_per_cm / self.px_per_cm
        size = (round(m.image.shape[1] * k), round(m.image.shape[0] * k))
        img = cv2.resize(m.image, size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(m.depth, size, interpolation=cv2.INTER_NEAREST)
        return m.image, img, depth

    def drop(self, bay_id: str) -> None:
        self._bays.pop(bay_id, None)
