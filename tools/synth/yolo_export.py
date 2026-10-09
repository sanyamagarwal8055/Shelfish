"""Synthetic pack-detection training data in YOLO format, weighted toward small packs.

    python -m tools.synth.yolo_export --out data/raw/synth_yolo --train 1500 --val 300 --seed 1

Shelves drawn from data/gallery photos with exact boxes (class 0 "object", like SKU-110K). Rows
are filled at random rather than from planograms, favouring small, tightly packed items (the
SKU-110K model misses soap bars, toothpaste and shampoo bottles): pack size jittered +-20%,
tight or loose spacing, random gaps, random shelf/rail colours, then a random crop/zoom,
brightness/contrast, blur and JPEG quality. Validation images use only held-out photos (one per
SKU with 2+ photos), so a validation score says whether the detector generalises to packaging it
never trained on. Writes images/{train,val}, labels/{train,val} and data.yaml (absolute paths).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from shelfpulse.config import REPO_ROOT
from shelfpulse.contracts import BAY_WIDTH_CM, ROWS, SkuRow, load_sku_master
from shelfpulse.perception.sku_index import gallery_photos, read_image
from tools.synth.make import BACKGROUND, PX_PER_CM, RAIL_CM, RAIL_COLOUR, load_geometry

SMALL_WIDTH_CM = 10.0  # packs at most this wide are "small" and drawn more often
SMALL_WEIGHT = 3.0
MIN_VISIBLE = 0.5  # after cropping, keep a box only if this share of it is still in view


@dataclass(frozen=True)
class Box:
    x0: float  # px
    y0: float
    x1: float
    y1: float


def split_photos(photos: dict[str, list[Path]]) -> tuple[dict, dict]:
    """(train photos, held-out val photos): the last photo of each SKU with 2+ photos is held out;
    SKUs with a single photo use it on both sides."""
    train, val = {}, {}
    for sku, ps in photos.items():
        train[sku], val[sku] = (ps[:-1], ps[-1:]) if len(ps) >= 2 else (ps, ps)
    return train, val


def _shift(colour, rng, amount: int) -> tuple[int, int, int]:
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in colour)


def render_shelf(
    shots: dict[str, list[np.ndarray]],
    skus: dict[str, SkuRow],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Box]]:
    """One random front-on bay at 10 px/cm and its pack boxes."""
    geo = load_geometry()
    h_px, w_px = geo.size_px
    bg, rail = _shift(BACKGROUND, rng, 30), _shift(RAIL_COLOUR, rng, 40)
    img = np.full((h_px, w_px, 3), bg, np.uint8)
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    names = sorted(s for s in shots if s in skus)
    weights = np.array([SMALL_WEIGHT if skus[s].width_cm <= SMALL_WIDTH_CM else 1.0 for s in names])
    weights /= weights.sum()
    boxes = []
    for r in ROWS:
        ry0, ry1 = geo.y_px(r * geo.pitch_cm + RAIL_CM), geo.y_px(r * geo.pitch_cm)
        img[ry0:ry1] = rail
        surface = ry0
        x = float(rng.uniform(0, 3))
        tight = rng.random() < 0.6
        while x < BAY_WIDTH_CM:
            sku = skus[names[rng.choice(len(names), p=weights)]]
            scale = float(rng.uniform(0.8, 1.2))
            run = int(rng.integers(2, 12))  # facings of the same SKU side by side
            for _ in range(run):
                w = sku.width_cm * scale
                if x + w > BAY_WIDTH_CM:
                    break
                full_h = sku.height_cm * scale
                h = min(full_h, geo.clearance_cm)
                if rng.random() > 0.08:  # ~8% of facings left empty (gaps)
                    w_px_, full_px, h_px_ = (
                        round(w * PX_PER_CM),
                        round(full_h * PX_PER_CM),
                        round(h * PX_PER_CM),
                    )
                    photo = shots[sku.sku_id][int(rng.integers(len(shots[sku.sku_id])))]
                    patch = cv2.resize(photo, (w_px_, full_px), interpolation=cv2.INTER_AREA)
                    patch = patch[full_px - h_px_ :]
                    x0 = round(x * PX_PER_CM)
                    img[surface - h_px_ : surface, x0 : x0 + w_px_] = patch[:, : w_px - x0]
                    boxes.append(Box(x0, surface - h_px_, min(x0 + w_px_, w_px), surface))
                x += w + (float(rng.uniform(0, 0.4)) if tight else float(rng.uniform(0.3, 3.0)))
            x += float(rng.uniform(0, 4))
    return img, boxes


def augment(
    img: np.ndarray, boxes: list[Box], rng: np.random.Generator, out_long: int = 1280
) -> tuple[np.ndarray, list[Box]]:
    """Random crop/zoom, resize, lighting, blur, JPEG; boxes cropped and scaled along."""
    h, w = img.shape[:2]
    ch = int(h * rng.uniform(0.35, 1.0))
    cw = int(w * rng.uniform(0.5, 1.0))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    img = img[y0 : y0 + ch, x0 : x0 + cw]
    kept = []
    for b in boxes:
        a = (b.x1 - b.x0) * (b.y1 - b.y0)
        cx0, cy0 = max(b.x0 - x0, 0), max(b.y0 - y0, 0)
        cx1, cy1 = min(b.x1 - x0, cw), min(b.y1 - y0, ch)
        if cx1 > cx0 and cy1 > cy0 and (cx1 - cx0) * (cy1 - cy0) >= MIN_VISIBLE * a:
            kept.append(Box(cx0, cy0, cx1, cy1))
    s = out_long / max(ch, cw) * rng.uniform(0.6, 1.0)
    img = cv2.resize(
        img, (max(1, round(cw * s)), max(1, round(ch * s))), interpolation=cv2.INTER_AREA
    )
    rh, rw = img.shape[:2]  # rounded size: clip so no box edge falls past the border
    kept = [
        Box(min(b.x0 * s, rw), min(b.y0 * s, rh), min(b.x1 * s, rw), min(b.y1 * s, rh))
        for b in kept
    ]
    img = cv2.convertScaleAbs(
        img, alpha=float(rng.uniform(0.7, 1.25)), beta=float(rng.uniform(-25, 25))
    )
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), kept


def write_split(out: Path, split: str, n: int, photos: dict[str, list[Path]], skus, rng) -> int:
    shots = {s: [read_image(p) for p in ps] for s, ps in photos.items()}
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(n):
        img, boxes = augment(*render_shelf(shots, skus, rng), rng)
        h, w = img.shape[:2]
        name = f"{split}_{i:05d}"
        cv2.imencode(".jpg", img)[1].tofile(str(out / "images" / split / f"{name}.jpg"))
        lines = [
            f"0 {(b.x0 + b.x1) / 2 / w:.6f} {(b.y0 + b.y1) / 2 / h:.6f} "
            f"{(b.x1 - b.x0) / w:.6f} {(b.y1 - b.y0) / h:.6f}"
            for b in boxes
        ]
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", "utf-8")
        total += len(boxes)
    return total


def export(out: Path, n_train: int, n_val: int, seed: int = 1, gallery: Path | None = None) -> dict:
    photos = gallery_photos(gallery or REPO_ROOT / "data" / "gallery")
    if not photos:
        raise ValueError("no gallery photos")
    skus = load_sku_master()
    train, val = split_photos(photos)
    rng = np.random.default_rng(seed)
    out = out.resolve()
    stats = {
        "train_images": n_train,
        "train_boxes": write_split(out, "train", n_train, train, skus, rng),
        "val_images": n_val,
        "val_boxes": write_split(out, "val", n_val, val, skus, rng),
    }
    data = {"path": str(out), "train": "images/train", "val": "images/val", "names": {0: "object"}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.yolo_export", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    ap.add_argument("--out", type=Path, default=Path("data/raw/synth_yolo"))
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        stats = export(args.out, args.train, args.val, args.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {stats} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
