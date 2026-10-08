"""Synthetic robot recordings: what the camera mast sees driving past bays.

    python -m tools.synth.robot --bays G3-L-05,G7-R-02,G9-E-F --out data/samples/robot/
        [--person G7-R-02] [--gaps] [--seed 1]

Per bay: the front-on bay (from its planogram, or a random one) rendered at ROBOT_PX_PER_CM with
shelf-edge price labels on the rails, and a depth map as the mast's ToF sees it (mm from the
camera, 0 = no reading). Frames are FRAME_WIDTH_CM-wide strips every FRAME_STEP_CM of travel.
Writes frames/*.jpg, depth/*.png, pose.csv (t, x_m, y_m, heading_deg, side, frame_file,
depth_file) and truth.jsonl (the true robot BayReading per bay, labels included). --person puts
a shopper in front of a bay, wide enough to cover more than 30% of it.

shelfpulse.sources.robot.vendor_api.FakeVendorAPI replays a recording, or calls render_pass()
directly for bays the recording lacks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from shelfpulse.contracts import (
    BayReading,
    Label,
    Planogram,
    SkuRow,
    load_sku_master,
    to_json_dict,
)
from shelfpulse.layout import store_map
from shelfpulse.sources.robot.geometry import Pose, bay_axis
from tools.synth import make as synth_make
from tools.synth.make import (
    CAMERA_MM,
    DEFAULT_START,
    PX_PER_CM,
    RAIL_CM,
    Geometry,
    Occluder,
    _blob_spans,
    _write,
    add_depth,
    add_gaps,
    layout,
    load_geometry,
    planograms_for,
    render,
    truth_reading,
)

ROBOT_PX_PER_CM = 20  # mast cameras shoot high-res; the stitcher scales to 10 px/cm for analyze
ROBOT_CAMERA_MM = 1000  # mast cameras ~1 m from the shelf edge
FRAME_WIDTH_CM = 60.0
FRAME_STEP_CM = 30.0
SPEED_MPS = 0.3
PERSON_WIDTH_CM = 60.0  # wide enough to hide > 30% of a 120 cm bay
PERSON_TOP_CM = 180.0


@dataclass
class RobotFrame:
    image: np.ndarray  # BGR, ROBOT_PX_PER_CM
    depth: np.ndarray  # uint16 mm from the camera
    pose: Pose


def price(sku: SkuRow) -> float:
    """Illustrative shelf price (same rule as the Brain's simulator)."""
    return round(sku.margin_inr * 5, 2)


def plan_labels(plano: Planogram, skus: dict[str, SkuRow]) -> dict[int, list[Label]]:
    """One shelf-edge label at the left edge of every planogram slot."""
    return {
        r: [Label(x_cm=s.x_start_cm, sku=s.sku_id, price=price(skus[s.sku_id])) for s in slots]
        for r, slots in enumerate(plano.rows)
    }


def draw_labels(img: np.ndarray, labels: dict[int, list[Label]], geo: Geometry, px: float):
    """White price tags on the rails: product id on top, price below."""
    k = px / PX_PER_CM
    for r, row in labels.items():
        y0 = round(geo.y_px(r * geo.pitch_cm + RAIL_CM) * k) + 2
        y1 = round(geo.y_px(r * geo.pitch_cm) * k) - 2
        for lb in row:
            x0 = round((lb.x_cm + 0.5) * px)
            x1 = min(x0 + round(9 * px), img.shape[1] - 1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (250, 250, 250), -1)
            scale = 0.022 * px
            font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
            name_y = y0 + round(0.45 * (y1 - y0))
            cv2.putText(img, lb.sku[:12], (x0 + 3, name_y), font, scale, (20, 20, 20), 1, aa)
            text = f"Rs {lb.price:.2f}"
            cv2.putText(img, text, (x0 + 3, y1 - 4), font, scale * 1.1, (20, 20, 160), 1, aa)


def render_pass(
    bay_id: str,
    plano: Planogram,
    skus: dict[str, SkuRow],
    rng: np.random.Generator,
    t0: datetime,
    *,
    gallery: dict | None = None,
    gaps: bool = False,
    person: bool = False,
    speed_mps: float = SPEED_MPS,
) -> tuple[list[RobotFrame], BayReading]:
    """Frames of one pass along a bay, and the true robot reading of that bay."""
    geo = load_geometry()
    packs = layout(plano, skus, geo)
    if gaps:
        packs = add_gaps(packs, rng)
    packs = add_depth(packs, skus, geo, rng)
    occ = None
    if person:
        x0 = round(float(rng.uniform(0, 120 - PERSON_WIDTH_CM)), 1)
        x1 = round(x0 + PERSON_WIDTH_CM, 1)
        occ = Occluder(x0, x1, PERSON_TOP_CM, _blob_spans(x0, x1, PERSON_TOP_CM, geo))
    img10, d10 = render(packs, occ, skus, geo, gallery or {}, rng, True)

    labels = plan_labels(plano, skus)
    truth = truth_reading(packs, occ, bay_id, "robot", t0, f"truth/{bay_id}.jpg")
    truth = truth.model_copy(
        update={
            "rows": [r.model_copy(update={"labels": labels.get(r.row, [])}) for r in truth.rows]
        }
    )
    BayReading.model_validate(to_json_dict(truth))  # still contract-valid with labels

    k = ROBOT_PX_PER_CM / PX_PER_CM
    h, w = round(img10.shape[0] * k), round(img10.shape[1] * k)
    img = cv2.resize(img10, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(d10, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    depth = np.where(depth > 0, depth - (CAMERA_MM - ROBOT_CAMERA_MM), 0).astype(np.uint16)
    draw_labels(img, labels, geo, ROBOT_PX_PER_CM)

    # Frames every FRAME_STEP_CM, FRAME_WIDTH_CM wide, cropped to the bay at its ends (the
    # neighbours aren't drawn, so no frame may show them); a frame's pose is its centre.
    axis = bay_axis(store_map.load().bays[bay_id])
    frames = []
    for i, c in enumerate(np.arange(0.0, 120.0 + 1e-6, FRAME_STEP_CM)):
        a = max(0.0, c - FRAME_WIDTH_CM / 2)
        b = min(120.0, c + FRAME_WIDTH_CM / 2)
        cols = slice(round(a * ROBOT_PX_PER_CM), round(b * ROBOT_PX_PER_CM))
        x_m, y_m = axis.pose_xy((a + b) / 2)
        t = t0 + timedelta(seconds=i * FRAME_STEP_CM / 100 / speed_mps)
        pose = Pose(t, round(x_m, 4), round(y_m, 4), axis.heading_deg, axis.side)
        frames.append(RobotFrame(img[:, cols].copy(), depth[:, cols].copy(), pose))
    return frames, truth


def record(
    out: Path,
    bays: list[str],
    seed: int = 1,
    *,
    people: set[str] = frozenset(),
    gaps: bool = False,
    planograms: Path | None = None,
    gallery: Path = Path("data/gallery"),
    start: datetime | None = None,
) -> list[BayReading]:
    rng = np.random.default_rng(seed)
    geo = load_geometry()
    skus = load_sku_master()
    planos = {p.bay_id: p for p in planograms_for(bays, planograms, skus, geo, rng)}
    shots = synth_make.load_gallery(gallery)  # via the module, so tests can switch it off
    t = start or datetime.fromisoformat(DEFAULT_START)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    rows, truths = [], []
    for bay in bays:
        frames, truth = render_pass(bay, planos[bay], skus, rng, t, gallery=shots, gaps=gaps,
                                    person=bay in people)  # fmt: skip
        for i, f in enumerate(frames):
            name = f"{bay}_{i:02d}"
            _write(out / "frames" / f"{name}.jpg", f.image)
            _write(out / "depth" / f"{name}.png", f.depth)
            p = f.pose
            rows.append([p.t.isoformat(), p.x_m, p.y_m, p.heading_deg, p.side,
                         f"frames/{name}.jpg", f"depth/{name}.png"])  # fmt: skip
        truths.append(truth)
        t = frames[-1].pose.t + timedelta(minutes=1)
    with open(out / "pose.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x_m", "y_m", "heading_deg", "side", "frame_file", "depth_file"])
        w.writerows(rows)
    with open(out / "truth.jsonl", "w", encoding="utf-8") as fh:
        for tr in truths:
            fh.write(json.dumps(to_json_dict(replace_ref(tr))) + "\n")
    return truths


def replace_ref(r: BayReading) -> BayReading:
    return r.model_copy(update={"frame_ref": f"truth/{r.bay_id}.jpg"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.robot", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    ap.add_argument("--bays", required=True, help="comma-separated bay_ids, in driving order")
    ap.add_argument("--out", type=Path, default=Path("data/samples/robot"))
    ap.add_argument("--person", default="", help="comma-separated bays with a shopper in front")
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gallery", type=Path, default=Path("data/gallery"))
    args = ap.parse_args(argv)
    people = {b for b in args.person.split(",") if b}
    try:
        truths = record(args.out, args.bays.split(","), args.seed, people=people, gaps=args.gaps,
                        gallery=args.gallery)  # fmt: skip
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote a robot pass over {len(truths)} bays to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
