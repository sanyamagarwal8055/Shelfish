"""Where the robot is versus which bay its camera sees.

The robot drives along a walkway centre line. Aisle bays are passed along y (heading north), end
caps along x on the front/back walkways (heading east). For each bay, contract x_cm (from the
bay's left edge as seen from the aisle, matching store_map.bay_x_to_store) is a linear function of
the robot's position along its path:
    x_cm = sign * (along - origin) * 100
and the bay is on the robot's left or right.
"""

from __future__ import annotations

from dataclasses import dataclass

from shelfpulse.contracts import BAY_WIDTH_CM
from shelfpulse.layout.store_map import Bay, StoreMap

LINE_TOLERANCE_M = 0.6  # robot must be this close to the bay's walkway centre line


@dataclass(frozen=True)
class Pose:
    t: object  # datetime
    x_m: float
    y_m: float
    heading_deg: float  # 90 = north (aisles), 0 = east (front/back walkways)
    side: str  # "left" | "right": which camera of the two-sided mast took the frame


@dataclass(frozen=True)
class BayAxis:
    bay_id: str
    axis: str  # "y" for aisle bays, "x" for end caps
    lo: float  # bay span along the path, m
    hi: float
    line: float  # the walkway centre line (the other coordinate), m
    heading_deg: float
    side: str
    origin: float  # path position where x_cm = 0
    sign: int  # +1 if x_cm grows along the path, -1 if it shrinks

    def x_cm(self, along: float) -> float:
        return self.sign * (along - self.origin) * 100.0

    def along(self, x_cm: float) -> float:
        return self.origin + self.sign * x_cm / 100.0

    def pose_xy(self, x_cm: float) -> tuple[float, float]:
        a = self.along(x_cm)
        return (self.line, a) if self.axis == "y" else (a, self.line)


def bay_axis(bay: Bay) -> BayAxis:
    if bay.kind == "bay":
        line = bay.front_xy[0]
        if bay.face == "L":  # bay east of the aisle: right of a robot heading north
            return BayAxis(bay.bay_id, "y", bay.y0, bay.y1, line, 90.0, "right", bay.y1, -1)
        return BayAxis(bay.bay_id, "y", bay.y0, bay.y1, line, 90.0, "left", bay.y0, +1)
    line = bay.front_xy[1]
    if bay.aisle_id == "front":  # cap north of the front walkway: left of a robot heading east
        return BayAxis(bay.bay_id, "x", bay.x0, bay.x1, line, 0.0, "left", bay.x0, +1)
    return BayAxis(bay.bay_id, "x", bay.x0, bay.x1, line, 0.0, "right", bay.x1, -1)


class BayLocator:
    """Bays a frame shows, from the robot's pose and the frame's width."""

    def __init__(self, smap: StoreMap):
        self.axes = [bay_axis(b) for b in smap.bays.values()]
        self.by_id = {a.bay_id: a for a in self.axes}

    def in_view(self, pose: Pose, width_cm: float) -> list[tuple[str, float]]:
        """[(bay_id, x_cm of the frame centre in that bay)] for every bay the frame overlaps."""
        out = []
        for a in self.axes:
            here, across = (pose.y_m, pose.x_m) if a.axis == "y" else (pose.x_m, pose.y_m)
            if a.side != pose.side or abs(across - a.line) > LINE_TOLERANCE_M:
                continue
            if abs((pose.heading_deg - a.heading_deg + 180) % 360 - 180) > 45:
                continue
            c = a.x_cm(here)
            if -width_cm / 2 < c < BAY_WIDTH_CM + width_cm / 2:
                out.append((a.bay_id, c))
        return out
