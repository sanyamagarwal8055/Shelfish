"""How many fixed cameras each camera-tier shelf face needs, and where they go.

A camera across the aisle at distance d sees a useful strip of width
    W = min(2d*tan(hfov/2), 2d*tan(max_angle), sensor_px / min_px_per_cm)
and neighbours share overlap_m, so a run of length L needs N = ceil((L - overlap) / (W - overlap))
cameras, centred at (i - 0.5) * L / N from the front.

One camera per position: the README's stacked second camera when the vertical view is shorter
than the shelf (README 5.1, `vertical_cover(d) < shelf_height`) is left out until the team gives
a vertical field of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shelfpulse.config import load_yaml
from shelfpulse.layout.store_map import StoreMap

EPS = 1e-9


@dataclass(frozen=True)
class PlanSettings:
    hfov_deg: float = 90.0
    sensor_px: int = 4000
    min_px_per_cm: float = 10.0
    max_angle_deg: float = 35.0
    overlap_m: float = 0.2


DEFAULT_SETTINGS = PlanSettings()


def load_settings(name: str = "perception") -> PlanSettings:
    return PlanSettings(**load_yaml(name)["camera_plan"])


@dataclass(frozen=True)
class Camera:
    id: str  # e.g. "G1-L-C1"
    run: str
    face: str
    aisle_id: str
    position_m: float  # along the run, from its front end
    view_m: tuple[float, float]  # strip of the face this camera sees, along the run
    mount: str  # run whose back it is fixed to across the aisle, or "pole"
    xy_m: tuple[float, float]  # store coordinates of the camera
    bays: tuple[str, ...]  # bays fully inside the view


@dataclass(frozen=True)
class FacePlan:
    run: str
    face: str
    aisle_id: str
    aisle_m: float
    width_m: float  # useful width W
    cameras: tuple[Camera, ...]
    unseen_bays: tuple[str, ...]  # bays no single camera sees whole (would need stitching)


def useful_width(d: float, s: PlanSettings = DEFAULT_SETTINGS) -> float:
    """Useful shelf width (m) one camera covers from distance d (m)."""
    return min(
        2 * d * math.tan(math.radians(s.hfov_deg / 2)),
        2 * d * math.tan(math.radians(s.max_angle_deg)),
        s.sensor_px / s.min_px_per_cm / 100,
    )


def positions(length: float, d: float, s: PlanSettings = DEFAULT_SETTINGS) -> list[float]:
    """Camera centres (m from the run's front) for one face of `length` m across a `d` m aisle."""
    w = useful_width(d, s)
    if w <= s.overlap_m:
        raise ValueError(f"aisle {d} m too narrow: useful width {w:.2f} m <= overlap")
    n = math.ceil((length - s.overlap_m) / (w - s.overlap_m) - EPS)
    return [round((i - 0.5) * length / n, 4) for i in range(1, n + 1)]


def _mount(smap: StoreMap, run_id: str, face: str) -> str:
    """The run across the aisle from this face (L faces west, R faces east), or a pole."""
    order = sorted(smap.runs.values(), key=lambda r: r.x_m)
    i = [r.id for r in order].index(run_id)
    j = i - 1 if face == "L" else i + 1
    return order[j].id if 0 <= j < len(order) else "pole"


def plan_face(
    smap: StoreMap, run_id: str, face: str, s: PlanSettings = DEFAULT_SETTINGS
) -> FacePlan:
    bays = smap.bays_of(run_id, face)
    run = smap.runs[run_id]
    d = smap.aisle_width_facing(bays[0].bay_id)
    w = useful_width(d, s)
    front_y = min(b.y0 for b in bays)
    plane_x = bays[0].x0 if face == "L" else bays[0].x1
    cam_x = plane_x - d if face == "L" else plane_x + d
    mount = _mount(smap, run_id, face)

    cams = []
    for i, p in enumerate(positions(run.length_m, d, s), start=1):
        lo, hi = p - w / 2, p + w / 2
        seen = tuple(
            b.bay_id for b in bays if b.y0 - front_y >= lo - EPS and b.y1 - front_y <= hi + EPS
        )
        cams.append(
            Camera(
                id=f"{run_id}-{face}-C{i}",
                run=run_id,
                face=face,
                aisle_id=bays[0].aisle_id,
                position_m=p,
                view_m=(round(lo, 4), round(hi, 4)),
                mount=mount,
                xy_m=(round(cam_x, 4), round(front_y + p, 4)),
                bays=seen,
            )
        )
    covered = {b for c in cams for b in c.bays}
    unseen = tuple(b.bay_id for b in bays if b.bay_id not in covered)
    return FacePlan(run_id, face, bays[0].aisle_id, d, round(w, 4), tuple(cams), unseen)


def plan(
    smap: StoreMap, runs: list[str] | None = None, s: PlanSettings = DEFAULT_SETTINGS
) -> list[FacePlan]:
    """Plan every face of the given runs (default: every camera-tier run)."""
    runs = runs or [r.id for r in smap.runs.values() if r.tier == "camera"]
    return [plan_face(smap, r, f, s) for r in runs for f in ("L", "R")]


def total_cameras(plans: list[FacePlan]) -> int:
    return sum(len(p.cameras) for p in plans)
