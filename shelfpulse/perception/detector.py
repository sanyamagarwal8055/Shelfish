"""Pack detection behind one small class, so the model can be swapped from config.

Backends:
- "yolo": an Ultralytics YOLO model (weights path from configs/perception.yaml), e.g. trained on
  SKU-110K with training/train_sku110k.py. Use this for real photos. With per_row, it runs on
  each shelf row's strip instead of the whole bay, so small packs keep enough pixels (a whole bay
  squeezed to 640 px makes a 6 cm soap ~18 px tall).
- "classic": OpenCV only. Packs are whatever differs from the shelf's back panel just above each
  shelf surface, split into facings at strong vertical edges. A baseline that works on synthetic
  shelves with no model; it is not meant for real photos.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shelfpulse.perception.shelves import Shelf

BACKENDS = ("classic", "yolo")


@dataclass(frozen=True)
class Box:
    x0: float  # px
    y0: float
    x1: float
    y1: float
    conf: float


@dataclass(frozen=True)
class ClassicSettings:
    bg_tol: float = 40.0  # colour distance from the back panel that counts as "something there"
    band_cm: tuple[float, float] = (1.0, 4.0)  # band above the shelf surface used to find packs
    min_pack_cm: float = 2.0
    edge_thr: float = 40.0  # Sobel |dI/dx| for a facing boundary
    edge_cover: float = 0.8  # fraction of band rows the boundary must span
    conf: float = 0.5


DEFAULT_CLASSIC = ClassicSettings()


def background_colour(img: np.ndarray, shelves: list[Shelf]) -> np.ndarray:
    """Most common colour in the upper part of every row, where the back panel usually shows."""
    parts = []
    for s in shelves:
        y1 = s.top + max(1, (s.surface - s.top) * 2 // 5)
        parts.append(img[s.top : y1].reshape(-1, 3))
    px = np.concatenate(parts) if parts else img.reshape(-1, 3)
    px = px[px.astype(np.int32).sum(1) > 60]  # skip black borders
    if px.size == 0:
        return np.zeros(3, np.float32)
    q = (px // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    mode = Counter(keys.tolist()).most_common(1)[0][0]
    return np.median(px[keys == mode], axis=0).astype(np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return [
        (int(r[0]), int(r[-1]) + 1) for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    ]


def _differs(img: np.ndarray, bg: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(img.astype(np.float32) - bg).max(-1) > tol


def classic_detect(
    img: np.ndarray, shelves: list[Shelf], px_per_cm: float, s: ClassicSettings = DEFAULT_CLASSIC
) -> list[Box]:
    bg = background_colour(img, shelves)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    min_px = round(s.min_pack_cm * px_per_cm)
    boxes = []
    for sh in shelves:
        y0 = max(sh.top, sh.surface - round(s.band_cm[1] * px_per_cm))
        y1 = max(y0 + 1, sh.surface - round(s.band_cm[0] * px_per_cm))
        occupied = _differs(img[y0:y1], bg, s.bg_tol).mean(0) >= 0.5
        boundary = ((gx[y0:y1] > s.edge_thr).mean(0) >= s.edge_cover) & occupied
        # One boundary per cluster of edge columns (a pack outline gives a pair of edges).
        cuts = [(a + b) // 2 for a, b in _runs(np.convolve(boundary, np.ones(9), "same") > 0)]
        strip = _differs(img[sh.top : sh.surface], bg, s.bg_tol)
        for a, b in _runs(occupied):
            inner = [c for c in cuts if a + min_px <= c <= b - min_px]
            for x0, x1 in zip([a, *inner], [*inner, b], strict=True):
                if x1 - x0 < min_px:
                    continue
                mid = strip[:, x0 + (x1 - x0) // 4 : x1 - (x1 - x0) // 4]
                filled = mid.mean(1) >= 0.5
                # Pack top: the highest row of the run of filled rows that touches the shelf.
                empty = np.flatnonzero(~filled[::-1])
                height = int(empty[0]) if empty.size else filled.size
                if height < min_px:
                    continue
                boxes.append(Box(x0, sh.surface - height, x1, sh.surface, s.conf))
    return boxes


class Detector:
    def __init__(
        self,
        backend: str = "classic",
        weights: str | Path | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        classic: ClassicSettings = DEFAULT_CLASSIC,
        per_row: bool = True,
        row_margin_cm: float = 2.0,
    ):
        if backend not in BACKENDS:  # "auto" is resolved by analyze.load_pipeline
            raise ValueError(f"detector backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.conf = conf
        self.imgsz = imgsz
        self.per_row = per_row
        self.row_margin_cm = row_margin_cm
        self.classic = classic
        self._model = None
        if backend == "yolo":
            if weights is None or not Path(weights).is_file():
                raise FileNotFoundError(
                    f"YOLO weights not found at {weights!r}. Train them with "
                    "training/train_sku110k.py (Colab/Kaggle) or get them from Sanyam's Drive, "
                    "then set detector.weights in configs/perception.yaml."
                )
            from ultralytics import YOLO  # heavy import, only when the YOLO backend is used

            self._model = YOLO(str(weights))

    def detect(self, img: np.ndarray, shelves: list[Shelf], px_per_cm: float) -> list[Box]:
        if self.backend == "classic":
            return classic_detect(img, shelves, px_per_cm, self.classic)
        if not (self.per_row and shelves):
            return self._yolo(img, 0)
        m = round(self.row_margin_cm * px_per_cm)
        boxes = []
        for s in shelves:
            y0, y1 = max(s.top - m, 0), min(s.surface + m, img.shape[0])
            boxes += self._yolo(img[y0:y1], y0)
        return boxes

    def _yolo(self, img: np.ndarray, y_off: int) -> list[Box]:
        res = self._model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [
            Box(float(b[0]), float(b[1]) + y_off, float(b[2]), float(b[3]) + y_off, float(c))
            for b, c in zip(xyxy, confs, strict=True)
        ]
