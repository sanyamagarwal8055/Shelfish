"""Synthetic camera time-lapse: shelves selling down, a shopper blocking a bay, a restock.

    python -m tools.synth.make --sequence 60 --out runs/seq01/ [--cameras G1-L-C2,G1-L-C3]

Every simulated minute each camera takes one frame of the two bays it sees (configs/cameras.yaml),
as a real shelf camera would: the bays side by side, seen with a slight downward tilt through a
wide lens. Writes, under --out:
    frames/<camera>/<minute>.jpg, depth/<camera>/<minute>.png   (depth: mm from the camera)
    frames.csv          t, camera, frame_file, depth_file
    calibration.yaml    per camera: lens k1 and the 4 corners of each bay in the undistorted
                        frame (what an installer measures once)
    truth.jsonl         the true BayReading of every bay every minute
    story.json          the scripted events

Default story (minutes from the start): RICE_1KG on G1-L-04 row 0 sells one pack a minute from
minute 5 until it is empty, staff restock G1-L-04 at minute 45; a shopper stands in front of
G1-L-05 from minute 30 to 50; every bay also sells slowly at random.
"""

from __future__ import annotations

import csv
import json
import math
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from shelfpulse.config import REPO_ROOT
from shelfpulse.contracts import BAY_WIDTH_CM, SkuRow, load_sku_master, to_json_dict
from shelfpulse.layout import store_map
from shelfpulse.sources.robot.geometry import bay_axis
from tools.synth import make as synth_make
from tools.synth.make import (
    DEFAULT_START,
    PX_PER_CM,
    Occluder,
    Placed,
    _blob_spans,
    _write,
    barrel,
    layout,
    load_geometry,
    planograms_for,
    render,
    truth_reading,
)

DEFAULT_CAMERAS = ["G1-L-C2", "G1-L-C3", "G3-L-C3"]
LENS_K1 = 0.12  # wide-lens barrel strength (same model as rectify.undistort_radial)
TILT_PX = 70  # top edge pulled in this much each side: camera mounted high, looking down
MARGIN_PX = 100  # frame border around the bays


@dataclass(frozen=True)
class Story:
    sell_out: tuple = ("G1-L-04", 0, 0, 5)  # bay, row, slot position, start minute (1 pack/min)
    restock: tuple = ("G1-L-04", 45)  # bay, minute
    occlude: tuple = ("G1-L-05", 30, 50, 30.0, 90.0)  # bay, from, to (excl.), x0_cm, x1_cm
    background_per_hour: float = 4.0  # random sales per bay per hour
    sell_per_minute: int = 1  # packs the sell_out slot loses each minute

    def as_dict(self) -> dict:
        return {
            "sell_out": dict(
                zip(("bay", "row", "position", "from_minute"), self.sell_out, strict=True)
            ),
            "restock": dict(zip(("bay", "minute"), self.restock, strict=True)),
            "occlude": dict(
                zip(("bay", "from", "to", "x0_cm", "x1_cm"), self.occlude, strict=True)
            ),
            "background_per_hour": self.background_per_hour,
            "sell_per_minute": self.sell_per_minute,
        }


@dataclass
class BayState:
    bay_id: str
    plano: object
    full: list[Placed]  # the bay when fully stocked
    packs: list[Placed] = field(default_factory=list)

    def restock(self) -> None:
        self.packs = list(self.full)

    def sell(self, i: int) -> None:
        p = self.packs[i]
        if p.depth_left and p.depth_left > 1:
            self.packs[i] = replace(p, depth_left=p.depth_left - 1)  # next pack moves up
        else:
            self.packs.pop(i)  # last one in that facing: the facing is gone


def _full(plano, skus: dict[str, SkuRow], geo) -> list[Placed]:
    return [
        replace(p, depth_left=max(1, math.floor(geo.shelf_depth_cm / skus[p.sku].depth_cm)))
        for p in layout(plano, skus, geo)
    ]


def left_to_right(bays: list[str]) -> list[str]:
    """Bays in the order a camera across the aisle sees them, left first."""
    smap = store_map.load()
    return sorted(bays, key=lambda b: bay_axis(smap.bays[b]).origin * bay_axis(smap.bays[b]).sign)


def _homography(w: int, h: int) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m, t = MARGIN_PX, TILT_PX
    dst = np.float32([[m + t, m], [m + w - t, m], [m + w, m + h], [m, m + h]])
    return cv2.getPerspectiveTransform(src, dst)


def _corners(H: np.ndarray, x0: int, x1: int, h: int) -> list[list[float]]:
    pts = np.float32([[[x0, 0]], [[x1, 0]], [[x1, h]], [[x0, h]]])
    return [
        [round(float(x), 2), round(float(y), 2)] for x, y in cv2.perspectiveTransform(pts, H)[:, 0]
    ]


def make_sequence(
    out: Path,
    minutes: int,
    seed: int = 1,
    cameras: list[str] | None = None,
    story: Story | None = None,
    gallery: Path = Path("data/gallery"),
    start: datetime | None = None,
) -> list:
    story = story or Story()
    rng = np.random.default_rng(seed)
    geo = load_geometry()
    skus = load_sku_master()
    plan = (REPO_ROOT / "configs" / "cameras.yaml").read_text(encoding="utf-8")
    cams = {c["id"]: c for c in yaml.safe_load(plan)["cameras"]}
    cameras = cameras or DEFAULT_CAMERAS
    unknown = [c for c in cameras if c not in cams]
    if unknown:
        raise ValueError(f"cameras not in configs/cameras.yaml: {unknown}")
    order = {c: left_to_right(cams[c]["bays"]) for c in cameras}
    bays = [b for c in cameras for b in order[c]]
    planos = {p.bay_id: p for p in planograms_for(bays, None, skus, geo, rng)}
    shots = synth_make.load_gallery(gallery)
    states = {b: BayState(b, planos[b], _full(planos[b], skus, geo)) for b in bays}
    for st in states.values():
        st.restock()

    bay_w, bay_h = round(BAY_WIDTH_CM * PX_PER_CM), geo.size_px[0]
    strip_w = bay_w * 2
    H = _homography(strip_w, bay_h)
    frame_size = (strip_w + 2 * MARGIN_PX, bay_h + 2 * MARGIN_PX)
    calibration = {
        c: {
            "k1": LENS_K1,
            "bays": {
                b: _corners(H, i * bay_w, (i + 1) * bay_w, bay_h) for i, b in enumerate(order[c])
            },
        }  # fmt: skip
        for c in cameras
    }
    for sub in ("frames", "depth"):
        for c in cameras:
            (out / sub / c).mkdir(parents=True, exist_ok=True)

    t0 = start or datetime.fromisoformat(DEFAULT_START)
    sell_bay, sell_row, sell_pos, sell_from = story.sell_out
    restock_bay, restock_min = story.restock
    occ_bay, occ_from, occ_to, occ_x0, occ_x1 = story.occlude
    slot = planos[sell_bay].rows[sell_row][sell_pos] if sell_bay in planos else None
    rows, truths = [], []
    for minute in range(minutes):
        t = t0 + timedelta(minutes=minute)
        # --- the minute's events -------------------------------------------------------------
        if slot is not None and minute >= sell_from:
            st = states[sell_bay]
            in_slot = [
                slot.x_start_cm <= p.x + p.w / 2 <= slot.x_end_cm and p.row == sell_row
                for p in st.packs
            ]
            for _ in range(story.sell_per_minute):
                idx = [i for i, ok in enumerate(in_slot) if ok]
                if not idx:
                    break
                n_before = len(st.packs)
                st.sell(idx[int(rng.integers(len(idx)))])
                if len(st.packs) < n_before:  # a facing emptied: recompute which packs are in it
                    in_slot = [
                        slot.x_start_cm <= p.x + p.w / 2 <= slot.x_end_cm and p.row == sell_row
                        for p in st.packs
                    ]
        for st in states.values():
            for _ in range(int(rng.poisson(story.background_per_hour / 60))):
                if st.packs:
                    st.sell(int(rng.integers(len(st.packs))))
        if restock_bay in states and minute == restock_min:
            states[restock_bay].restock()

        # --- frames --------------------------------------------------------------------------
        for c in cameras:
            imgs, depths = [], []
            for b in order[c]:
                occ = None
                if b == occ_bay and occ_from <= minute < occ_to:
                    occ = Occluder(occ_x0, occ_x1, 175.0, _blob_spans(occ_x0, occ_x1, 175.0, geo))
                bay_rng = np.random.default_rng(zlib.crc32(f"{seed}:{b}".encode()))  # same photos
                img, dmap = render(states[b].packs, occ, skus, geo, shots, bay_rng, True)
                imgs.append(img)
                depths.append(dmap)
                ref = f"frames/{c}/{minute:04d}.jpg#{b}"
                truths.append(truth_reading(states[b].packs, occ, b, "camera", t, ref))
            strip, dstrip = np.hstack(imgs), np.hstack(depths)
            frame = cv2.warpPerspective(strip, H, frame_size, flags=cv2.INTER_LINEAR)
            dframe = cv2.warpPerspective(dstrip, H, frame_size, flags=cv2.INTER_NEAREST)
            frame, dframe = barrel(frame, LENS_K1), barrel(dframe, LENS_K1, nearest=True)
            name = f"{minute:04d}"
            _write(out / "frames" / c / f"{name}.jpg", frame)
            _write(out / "depth" / c / f"{name}.png", dframe)
            rows.append([t.isoformat(), c, f"frames/{c}/{name}.jpg", f"depth/{c}/{name}.png"])

    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "camera", "frame_file", "depth_file"])
        w.writerows(rows)
    (out / "calibration.yaml").write_text(
        "# Synthetic install-time calibration: lens k1 + bay corners (TL, TR, BR, BL) in the\n"
        "# undistorted frame, per camera.\n"
        + yaml.safe_dump({"cameras": calibration}, sort_keys=False),
        encoding="utf-8",
    )
    (out / "story.json").write_text(json.dumps(story.as_dict(), indent=2), encoding="utf-8")
    with open(out / "truth.jsonl", "w", encoding="utf-8") as f:
        for tr in truths:
            f.write(json.dumps(to_json_dict(tr)) + "\n")
    return truths
