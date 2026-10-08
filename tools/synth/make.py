"""Synthetic front-on bay images with perfect ground truth.

    python -m tools.synth.make --n 50 --seed 1 --out runs/synth01/ [--gaps] [--misplace]
        [--occlude] [--distort] [--depth]
    python -m tools.synth.make --sequence 60 --out runs/seq01/   (camera time-lapse, sequence.py)

Each image is drawn at 10 px/cm (1200 x 2100 px, 6 rows, row 0 at the bottom) from a bay's
planogram, using pack sizes from data/sku_master.csv. Packs are pasted from
data/gallery/<sku_id>/*.jpg, or drawn as coloured rectangles with the SKU name if the gallery is
empty. Because the generator places every pack itself, it writes the true BayReading for each
image to truth.jsonl.

Packs taller than the shelf clearance are cut off at the shelf above, as a front-on camera sees
them; the truth's h_cm is the visible height. --distort bends the image only: the truth stays the
front-on answer that rectify must recover.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from shelfpulse.config import load_yaml
from shelfpulse.contracts import (
    BAY_WIDTH_CM,
    MIN_GAP_CM,
    ROWS,
    BayReading,
    Gap,
    Pack,
    Planogram,
    PlanogramSlot,
    Row,
    SkuRow,
    load_sku_master,
    parse_planogram,
    to_json_dict,
)
from shelfpulse.layout import store_map

PX_PER_CM = 10.0
RAIL_CM = 3.0  # shelf-edge rail at the bottom of each row
BACKGROUND = (205, 210, 215)  # BGR shelf back panel
RAIL_COLOUR = (60, 85, 120)
PERSON_COLOUR = (55, 45, 40)
DISTORT_K = 0.12  # barrel strength
CAMERA_MM = 2000  # depth maps: a camera 2 m across the aisle from the shelf edge
PERSON_MM = 500  # a person stands this far in front of the shelf edge
DEFAULT_START = "2026-10-08T09:00:00+05:30"


@dataclass(frozen=True)
class Geometry:
    height_cm: float
    shelf_depth_cm: float
    pitch_cm: float  # height of one row, rail included

    @property
    def clearance_cm(self) -> float:
        return self.pitch_cm - RAIL_CM

    def floor_cm(self, row: int) -> float:
        """Height of row `row`'s shelf surface above the bay bottom."""
        return row * self.pitch_cm + RAIL_CM

    @property
    def size_px(self) -> tuple[int, int]:
        return round(self.height_cm * PX_PER_CM), round(BAY_WIDTH_CM * PX_PER_CM)

    def y_px(self, y_cm: float) -> int:
        """Height above the bay bottom (cm) -> image row (px, 0 at the top)."""
        return round((self.height_cm - y_cm) * PX_PER_CM)


def load_geometry() -> Geometry:
    bay = load_yaml("store_layout")["bay"]
    height = float(bay["height_m"]) * 100
    return Geometry(height, float(bay["depth_cm"]), height / int(bay["levels"]))


@dataclass(frozen=True)
class Placed:
    row: int
    sku: str
    x: float  # left edge, cm
    w: float
    h: float  # visible height, cm
    depth_left: int | None = None


@dataclass(frozen=True)
class Occluder:
    x0: float
    x1: float
    top_cm: float  # height of the head's top above the bay bottom
    spans: tuple[tuple[int, float, float], ...]  # (row, x0, x1): what the blob covers per row


# --------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------


def layout(plano: Planogram, skus: dict[str, SkuRow], geo: Geometry) -> list[Placed]:
    """Fill every planogram slot with its facings, spread evenly across the slot."""
    out = []
    for r, slots in enumerate(plano.rows):
        for s in slots:
            sku = skus[s.sku_id]
            slot_w = s.x_end_cm - s.x_start_cm
            n = min(s.facings, math.floor(slot_w / sku.width_cm + 1e-9))
            if n <= 0:
                continue
            space = (slot_w - n * sku.width_cm) / (n + 1)
            h = min(sku.height_cm, geo.clearance_cm)
            for k in range(n):
                x = s.x_start_cm + space + k * (sku.width_cm + space)
                out.append(Placed(r, s.sku_id, round(x, 2), sku.width_cm, h))
    return out


def add_gaps(packs: list[Placed], rng: np.random.Generator) -> list[Placed]:
    """Remove a run of 1-3 neighbouring packs from about half the rows (at least one)."""
    by_row: dict[int, list[Placed]] = {}
    for p in packs:
        by_row.setdefault(p.row, []).append(p)
    rows = sorted(by_row)
    chosen = [r for r in rows if rng.random() < 0.5] or [rows[rng.integers(len(rows))]]
    drop: set[Placed] = set()
    for r in chosen:
        row = sorted(by_row[r], key=lambda p: p.x)
        n = int(rng.integers(1, min(3, len(row)) + 1))
        start = int(rng.integers(0, len(row) - n + 1))
        drop.update(row[start : start + n])
    return [p for p in packs if p not in drop]


def misplace(
    packs: list[Placed], skus: dict[str, SkuRow], geo: Geometry, rng: np.random.Generator
) -> list[Placed]:
    """Swap one pack for a product that doesn't belong in its row and is no wider than it."""
    order = rng.permutation(len(packs))
    for i in order:
        p = packs[i]
        in_row = {q.sku for q in packs if q.row == p.row}
        cands = sorted(s for s, row in skus.items() if s not in in_row and row.width_cm <= p.w)
        if cands:
            new = skus[cands[int(rng.integers(len(cands)))]]
            h = min(new.height_cm, geo.clearance_cm)
            packs = list(packs)
            packs[i] = replace(p, sku=new.sku_id, w=new.width_cm, h=h)
            return packs
    return packs


def add_depth(
    packs: list[Placed], skus: dict[str, SkuRow], geo: Geometry, rng: np.random.Generator
) -> list[Placed]:
    out = []
    for p in packs:
        fit = max(1, math.floor(geo.shelf_depth_cm / skus[p.sku].depth_cm))
        out.append(replace(p, depth_left=int(rng.integers(1, fit + 1))))
    return out


def recess_cm(p: Placed, skus: dict[str, SkuRow], geo: Geometry) -> float:
    """How far the front pack sits back from the shelf edge: remaining packs are at the back."""
    return max(0.0, geo.shelf_depth_cm - (p.depth_left or 0) * skus[p.sku].depth_cm)


def make_occluder(geo: Geometry, rng: np.random.Generator) -> Occluder:
    w = float(rng.uniform(30, 50))
    x0 = round(float(rng.uniform(0, BAY_WIDTH_CM - w)), 1)
    top = float(rng.uniform(120, 180))
    return Occluder(x0, round(x0 + w, 1), top, _blob_spans(x0, round(x0 + w, 1), top, geo))


def _blob_spans(x0: float, x1: float, top: float, geo: Geometry) -> tuple:
    """Per row, the x range the blob (body + round head, as render draws it) covers."""
    rad = (round(x1 * PX_PER_CM) - round(x0 * PX_PER_CM)) // 3 / PX_PER_CM
    cx, cy = (x0 + x1) / 2, top - rad  # head centre (height above the bay bottom)
    body_top = top - 2 * rad
    spans = []
    for r in ROWS:
        lo, hi = r * geo.pitch_cm, (r + 1) * geo.pitch_cm
        if lo < body_top:
            spans.append((r, x0, x1))
        elif lo < top:
            dy = 0.0 if lo <= cy <= hi else min(abs(cy - lo), abs(cy - hi))
            half = math.sqrt(max(rad * rad - dy * dy, 0.0))
            if half > 0:
                spans.append((r, round(max(cx - half, 0.0), 1), round(min(cx + half, 120.0), 1)))
    return tuple(spans)


# --------------------------------------------------------------------------------------------
# Truth
# --------------------------------------------------------------------------------------------


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def free_stretches(
    taken: list[tuple[float, float]], width: float = BAY_WIDTH_CM
) -> list[tuple[float, float]]:
    """Intervals of [0, width] not covered by `taken`."""
    out, x = [], 0.0
    for a, b in sorted(taken):
        if a > x:
            out.append((x, a))
        x = max(x, b)
    if x < width:
        out.append((x, width))
    return out


def truth_reading(
    packs: list[Placed],
    occ: Occluder | None,
    bay_id: str,
    source: str,
    t: datetime,
    frame_ref: str,
) -> BayReading:
    rows = []
    for r in ROWS:
        hidden = [(a, b) for row, a, b in occ.spans if row == r] if occ else []
        # A pack more than half hidden can't be seen, so it isn't in the truth.
        in_row = sorted((p for p in packs if p.row == r), key=lambda p: p.x)
        seen = [
            p
            for p in in_row
            if not any(_overlap(p.x, p.x + p.w, a, b) > p.w / 2 for a, b in hidden)
        ]
        # The visible sliver of a mostly hidden pack is still not empty shelf, so not a gap.
        taken = [(p.x, p.x + p.w) for p in in_row] + hidden
        stretches = [(round(a, 2), round(b, 2)) for a, b in free_stretches(taken)]
        gaps = [
            Gap(x_cm=a, w_cm=round(b - a, 2)) for a, b in stretches if round(b - a, 2) > MIN_GAP_CM
        ]
        rows.append(
            Row(
                row=r,
                occluded=[list(h) for h in hidden],
                packs=[
                    Pack(
                        sku=p.sku,
                        conf=1.0,
                        x_cm=p.x,
                        w_cm=p.w,
                        h_cm=p.h,
                        stack=1,
                        depth_left=p.depth_left,
                    )
                    for p in seen
                ],
                gaps=gaps,
            )
        )
    return BayReading(
        bay_id=bay_id,
        source=source,
        t=t,
        frame_ref=frame_ref,
        quality=1.0,
        px_per_cm=PX_PER_CM,
        rows=rows,
    )


# --------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------


def sku_colour(sku: str) -> tuple[int, int, int]:
    hue = zlib.crc32(sku.encode()) % 180
    hsv = np.uint8([[[hue, 150, 200]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _read(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gallery(root: Path) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [_read(f) for f in sorted(d.glob("*.jpg"))]
        imgs = [im for im in imgs if im is not None]
        if imgs:
            out[d.name] = imgs
    return out


def pack_patch(
    p: Placed,
    full_h: float,
    gallery: dict[str, list[np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    w_px, h_px = round(p.w * PX_PER_CM), round(p.h * PX_PER_CM)
    full_px = round(full_h * PX_PER_CM)
    shots = gallery.get(p.sku)
    if shots:
        img = cv2.resize(shots[int(rng.integers(len(shots)))], (w_px, full_px))
        return img[full_px - h_px :]  # cut off at the shelf above, keep the bottom
    patch = np.full((h_px, w_px, 3), sku_colour(p.sku), np.uint8)
    cv2.rectangle(patch, (0, 0), (w_px - 1, h_px - 1), (30, 30, 30), 2)
    # Narrow packs get their name written vertically.
    vertical = w_px < 120
    canvas = np.ascontiguousarray(np.rot90(patch)) if vertical else patch
    ch, cw = canvas.shape[:2]
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(p.sku, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    org = (max(2, (cw - tw) // 2), (ch + th) // 2)
    cv2.putText(canvas, p.sku, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return np.ascontiguousarray(np.rot90(canvas, -1)) if vertical else canvas


def render(
    packs: list[Placed],
    occ: Occluder | None,
    skus: dict[str, SkuRow],
    geo: Geometry,
    gallery: dict[str, list[np.ndarray]],
    rng: np.random.Generator,
    with_depth: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    h_px, w_px = geo.size_px
    img = np.full((h_px, w_px, 3), BACKGROUND, np.uint8)
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Depth like a stereo camera: mm from the camera (0 = no reading). Rails sit at CAMERA_MM,
    # the empty shelf's back panel shelf_depth further, a pack front at its recess.
    back = CAMERA_MM + round(geo.shelf_depth_cm * 10)
    depth = np.full((h_px, w_px), back, np.uint16) if with_depth else None

    for r in ROWS:
        y0, y1 = geo.y_px(r * geo.pitch_cm + RAIL_CM), geo.y_px(r * geo.pitch_cm)
        img[y0:y1] = RAIL_COLOUR
        if depth is not None:
            depth[y0:y1] = CAMERA_MM

    for p in packs:
        x0 = round(p.x * PX_PER_CM)
        y1 = geo.y_px(geo.floor_cm(p.row))
        patch = pack_patch(p, skus[p.sku].height_cm, gallery, rng)
        ph, pw = patch.shape[:2]
        img[y1 - ph : y1, x0 : x0 + pw] = patch[:, : w_px - x0]
        if depth is not None:
            depth[y1 - ph : y1, x0 : x0 + pw] = CAMERA_MM + round(recess_cm(p, skus, geo) * 10)

    if occ is not None:
        x0, x1 = round(occ.x0 * PX_PER_CM), round(occ.x1 * PX_PER_CM)
        head_r = (x1 - x0) // 3
        head_top = geo.y_px(occ.top_cm)
        body_top = head_top + 2 * head_r
        cv2.rectangle(img, (x0, body_top), (x1, h_px - 1), PERSON_COLOUR, -1)
        cv2.circle(img, ((x0 + x1) // 2, head_top + head_r), head_r, PERSON_COLOUR, -1)
        if depth is not None:
            near = CAMERA_MM - PERSON_MM
            cv2.rectangle(depth, (x0, body_top), (x1, h_px - 1), near, -1)
            cv2.circle(depth, ((x0 + x1) // 2, head_top + head_r), head_r, near, -1)
    return img, depth


def barrel(img: np.ndarray, k: float = DISTORT_K, nearest: bool = False) -> np.ndarray:
    """Wide-lens barrel distortion: straight shelf rails bow outwards."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    s = math.hypot(cx, cy)
    yy, xx = np.indices((h, w), dtype=np.float32)
    nx, ny = (xx - cx) / s, (yy - cy) / s
    f = 1 + k * (nx * nx + ny * ny)
    map_x, map_y = (cx + nx * f * s).astype(np.float32), (cy + ny * f * s).astype(np.float32)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(img, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT)


def _write(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def load_planograms(folder: Path) -> list[Planogram]:
    planos = [
        parse_planogram(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(folder.glob("*.json"))
    ]
    if not planos:
        raise ValueError(f"no planograms in {folder}")
    return planos


def random_planogram(
    bay_id: str, skus: dict[str, SkuRow], geo: Geometry, rng: np.random.Generator
) -> Planogram:
    """A plausible planogram for a bay that has none: 1-3 SKUs per row from the run's category
    (any SKU if the category has none), each slot as wide as a whole number of facings."""
    smap = store_map.load()
    category = smap.runs[smap.bays[bay_id].run].category
    pool = sorted(s for s, r in skus.items() if r.category == category) or sorted(skus)
    rows = []
    for _ in ROWS:
        n = int(rng.integers(1, 4))
        picks = [pool[int(i)] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
        bounds = np.linspace(0, BAY_WIDTH_CM, len(picks) + 1)
        slots = []
        for pos, (sku, a, b) in enumerate(zip(picks, bounds, bounds[1:], strict=False)):
            facings = int((b - a) // skus[sku].width_cm)
            slots.append(
                PlanogramSlot(
                    position=pos,
                    sku_id=sku,
                    x_start_cm=round(float(a), 2),
                    x_end_cm=round(float(b), 2),
                    facings=facings,
                    min_facings=min(1, facings),
                )
            )
        rows.append(slots)
    return Planogram(bay_id=bay_id, source="planogram", rows=rows)


def planograms_for(
    bays: list[str] | None,
    folder: Path | None,
    skus: dict[str, SkuRow],
    geo: Geometry,
    rng: np.random.Generator,
) -> list[Planogram]:
    """Planograms from the folder; bays named but not in it get a random_planogram."""
    planos = load_planograms(folder or default_planograms())
    if not bays:
        return planos
    by_bay = {p.bay_id: p for p in planos}
    smap = store_map.load()
    unknown = [b for b in bays if b not in smap.bays]
    if unknown:
        raise ValueError(f"not bays in configs/store_layout.yaml: {unknown}")
    return [by_bay.get(b) or random_planogram(b, skus, geo, rng) for b in bays]


def default_planograms() -> Path:
    real = Path("data/planograms")
    return (
        real
        if real.is_dir() and any(real.glob("*.json"))
        else Path("contracts/fixtures/demo_store/planograms")
    )


def make(
    out: Path,
    n: int,
    seed: int,
    *,
    gaps: bool = False,
    misplaced: bool = False,
    occlude: bool = False,
    distort: bool = False,
    depth: bool = False,
    planograms: Path | None = None,
    sku_master: Path = Path("data/sku_master.csv"),
    gallery: Path = Path("data/gallery"),
    bays: list[str] | None = None,
    start: datetime | None = None,
) -> list[BayReading]:
    """Write n images (+ depth maps) and truth.jsonl to `out`. Returns the truth readings."""
    rng = np.random.default_rng(seed)
    geo = load_geometry()
    skus = load_sku_master(sku_master)
    smap = store_map.load()
    planos = planograms_for(bays, planograms, skus, geo, rng)
    shots = load_gallery(gallery)
    t0 = start or datetime.fromisoformat(DEFAULT_START)

    (out / "images").mkdir(parents=True, exist_ok=True)
    if depth:
        (out / "depth").mkdir(exist_ok=True)

    truths = []
    for i in range(n):
        plano = planos[i % len(planos)]
        packs = layout(plano, skus, geo)
        if misplaced:
            packs = misplace(packs, skus, geo, rng)
        if gaps:
            packs = add_gaps(packs, rng)
        if depth:
            packs = add_depth(packs, skus, geo, rng)
        occ = make_occluder(geo, rng) if occlude else None

        img, dmap = render(packs, occ, skus, geo, shots, rng, depth)
        if distort:
            img = barrel(img)
            dmap = barrel(dmap, nearest=True) if dmap is not None else None

        name = f"{i:04d}_{plano.bay_id}"
        _write(out / "images" / f"{name}.jpg", img)
        if dmap is not None:
            _write(out / "depth" / f"{name}.png", dmap)

        source = "camera" if smap.bays[plano.bay_id].tier == "camera" else "robot"
        t = t0 + timedelta(minutes=i)
        truths.append(truth_reading(packs, occ, plano.bay_id, source, t, f"images/{name}.jpg"))

    with open(out / "truth.jsonl", "w", encoding="utf-8") as f:
        for tr in truths:
            f.write(json.dumps(to_json_dict(tr)) + "\n")
    return truths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.make", description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gaps", action="store_true", help="remove runs of packs")
    ap.add_argument("--misplace", action="store_true", help="swap one pack per bay")
    ap.add_argument("--occlude", action="store_true", help="person-shaped blob")
    ap.add_argument("--distort", action="store_true", help="wide-lens barrel distortion")
    ap.add_argument("--depth", action="store_true", help="write depth maps + depth_left")
    ap.add_argument("--planograms", type=Path, help="folder of <bay_id>.json")
    ap.add_argument("--sku-master", type=Path, default=Path("data/sku_master.csv"))
    ap.add_argument("--gallery", type=Path, default=Path("data/gallery"))
    ap.add_argument(
        "--bays", help="comma-separated bay_ids (default: every planogram; others get a random one)"
    )
    ap.add_argument("--start", type=datetime.fromisoformat, help="ISO time of the first frame")
    ap.add_argument(
        "--sequence", type=int, metavar="N", help="camera time-lapse: N minutes (see sequence.py)"
    )
    ap.add_argument("--cameras", help="--sequence: comma-separated camera ids")
    args = ap.parse_args(argv)
    if args.sequence:
        from tools.synth.sequence import make_sequence

        try:
            cams = args.cameras.split(",") if args.cameras else None
            truths = make_sequence(args.out, args.sequence, args.seed, cams, None, args.gallery,
                                   args.start)  # fmt: skip
        except (ValueError, KeyError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"wrote a {args.sequence}-minute camera sequence ({len(truths)} bay readings) "
              f"to {args.out}")  # fmt: skip
        return 0
    try:
        truths = make(
            args.out,
            args.n,
            args.seed,
            gaps=args.gaps,
            misplaced=args.misplace,
            occlude=args.occlude,
            distort=args.distort,
            depth=args.depth,
            planograms=args.planograms,
            sku_master=args.sku_master,
            gallery=args.gallery,
            bays=args.bays.split(",") if args.bays else None,
            start=args.start,
        )
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {len(truths)} images and truth.jsonl to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
