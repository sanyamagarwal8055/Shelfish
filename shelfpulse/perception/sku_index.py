"""FAISS search over gallery pack-shots: crop embedding -> closest SKUs.

The gallery has only 1-3 photos per SKU, so each photo is indexed in several variants that mimic
how a pack looks on a shelf image: squeezed to the pack's real width:height, cut off at the shelf
above, cropped, darker/brighter, slightly blurred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import faiss
import numpy as np

from shelfpulse.contracts import SkuRow
from shelfpulse.perception.embedder import Embedder

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def gallery_photos(root: Path) -> dict[str, list[Path]]:
    """{sku_id: photo paths} from root/<sku_id>/*.jpg."""
    out: dict[str, list[Path]] = {}
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            photos = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if photos:
                out[d.name] = photos
    return out


def variants(img: np.ndarray, sku: SkuRow | None, clearance_cm: float = 32.0) -> list[np.ndarray]:
    """The photo plus shelf-like versions of it."""
    h, w = img.shape[:2]
    out = [img]
    if sku is not None:  # as a front-on shelf shows it: real aspect, top cut by the shelf above
        px = 10
        full = cv2.resize(img, (round(sku.width_cm * px), round(sku.height_cm * px)))
        vis = min(sku.height_cm, clearance_cm)
        out.append(full[full.shape[0] - round(vis * px) :])
    out.append(img[h // 12 : h - h // 12, w // 12 : w - w // 12])  # tighter crop
    out.append(img[h // 4 :])  # lower three quarters (front partly hidden)
    out.append(cv2.convertScaleAbs(img, alpha=0.7, beta=0))  # darker
    out.append(cv2.convertScaleAbs(img, alpha=1.2, beta=15))  # brighter
    out.append(cv2.GaussianBlur(img, (5, 5), 0))
    return out


@dataclass
class SkuIndex:
    index: faiss.Index
    labels: list[str]  # sku_id per indexed vector
    model: str  # embedder name it was built with

    @classmethod
    def build(
        cls,
        photos: dict[str, list[Path]],
        embedder: Embedder,
        skus: dict[str, SkuRow] | None = None,
        clearance_cm: float = 32.0,
    ) -> SkuIndex:
        crops, labels = [], []
        for sku_id, paths in sorted(photos.items()):
            for p in paths:
                img = read_image(p)
                if img is None:
                    raise ValueError(f"cannot read gallery photo {p}")
                vs = variants(img, (skus or {}).get(sku_id), clearance_cm)
                crops += vs
                labels += [sku_id] * len(vs)
        if not crops:
            raise ValueError("gallery is empty")
        vecs = embedder.embed(crops)
        index = faiss.IndexFlatIP(vecs.shape[1])  # inner product of unit vectors = cosine
        index.add(vecs)
        return cls(index, labels, embedder.name)

    def search(self, vecs: np.ndarray, k: int = 5) -> list[list[tuple[str, float]]]:
        """Per query: up to k (sku_id, cosine) pairs, best first, one entry per SKU."""
        sims, ids = self.index.search(vecs.astype(np.float32), min(len(self.labels), k * 8))
        out = []
        for row_s, row_i in zip(sims, ids, strict=True):
            best: dict[str, float] = {}
            for s, i in zip(row_s, row_i, strict=True):
                if i >= 0 and self.labels[i] not in best:
                    best[self.labels[i]] = float(s)
            out.append(sorted(best.items(), key=lambda kv: -kv[1])[:k])
        return out

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        # serialize to bytes ourselves: faiss' own file writer can't open non-ASCII Windows paths
        (folder / "index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())
        meta = {"model": self.model, "labels": self.labels}
        (folder / "labels.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> SkuIndex:
        raw = np.frombuffer((folder / "index.faiss").read_bytes(), dtype=np.uint8)
        meta = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        return cls(faiss.deserialize_index(raw), meta["labels"], meta["model"])
