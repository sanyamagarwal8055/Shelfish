"""The store as data (shared): runs, faces, bays, aisles and a walkable aisle graph.

Pure geometry from configs/store_layout.yaml. No routing policy here: the Brain's robot planner
orders bays; the Vision camera planner places cameras. Both read this map.

Coordinates are metres, origin at the store's front-left corner, x east, y towards the back.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path

from shelfpulse.config import load_yaml
from shelfpulse.contracts import BAY_WIDTH_CM, BayIdParts, parse_bay_id

__all__ = ["Aisle", "Bay", "BayIdParts", "Run", "StoreMap", "load", "parse_bay_id"]

DOCK = "DOCK"
DOOR = "DOOR"


@dataclass(frozen=True)
class Run:
    id: str
    x_m: float  # west edge
    y_m: float  # front edge
    length_m: float
    depth_m: float
    tier: str  # "camera" | "robot_only"
    category: str


@dataclass(frozen=True)
class Aisle:
    id: str
    x_m: float  # centre line
    width_m: float


@dataclass(frozen=True)
class Bay:
    bay_id: str
    run: str
    face: str  # "L", "R" or "E"
    index: int | None  # 1..10 for aisle bays, None for end caps
    kind: str  # "bay" | "end_cap"
    x0: float  # footprint, metres
    x1: float
    y0: float
    y1: float
    aisle_id: str  # "A0".."A10", or "front" / "back" walkway for end caps
    front_xy: tuple[float, float]  # point on the walkway centre line in front of the bay
    tier: str


@dataclass
class StoreMap:
    runs: dict[str, Run]
    aisles: dict[str, Aisle]
    bays: dict[str, Bay]
    cross_aisles: dict[str, dict]
    dock_xy: tuple[float, float]
    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    edges: dict[str, dict[str, float]] = field(default_factory=dict)

    # --- lookups -----------------------------------------------------------------------------

    def tier(self, run_id: str) -> str:
        return self.runs[run_id].tier

    def bays_of(self, run_id: str, face: str) -> list[Bay]:
        """Bays on one face of a run, front to back (or the two end caps for face "E")."""
        out = [b for b in self.bays.values() if b.run == run_id and b.face == face]
        return sorted(out, key=lambda b: (b.index or 0, b.bay_id))

    def camera_bays(self) -> list[str]:
        """Aisle bays on camera-tier runs (end caps are robot-only)."""
        return [b.bay_id for b in self.bays.values() if b.tier == "camera" and b.kind == "bay"]

    def aisle_width_facing(self, bay_id: str) -> float:
        b = self.bays[bay_id]
        if b.kind == "end_cap":
            return float(self.cross_aisles[b.aisle_id]["width_m"])
        return self.aisles[b.aisle_id].width_m

    def bay_x_to_store(self, bay_id: str, x_cm: float) -> tuple[float, float]:
        """Contract x_cm (from the bay's left edge, seen from the aisle) -> store (x_m, y_m).

        Face L is seen looking east, so its left edge is the back (+y) end. Face R is seen
        looking west: left edge at the front. Front end caps are seen looking north (left =
        west); back end caps looking south (left = east).
        """
        if not 0.0 <= x_cm <= BAY_WIDTH_CM:
            raise ValueError(f"x_cm {x_cm} outside 0..{BAY_WIDTH_CM}")
        b = self.bays[bay_id]
        d = x_cm / 100.0
        if b.face == "L":
            return (b.x0, b.y1 - d)
        if b.face == "R":
            return (b.x1, b.y0 + d)
        if b.aisle_id == "front":
            return (b.x0 + d, b.y0)
        return (b.x1 - d, b.y1)

    # --- aisle graph -------------------------------------------------------------------------

    def shortest_path(self, a: str, b: str) -> tuple[float, list[str]]:
        """Walking distance (m) and node path between two graph nodes (bay ids, DOCK, ...)."""
        dist = {a: 0.0}
        prev: dict[str, str] = {}
        heap = [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                path = [b]
                while path[-1] != a:
                    path.append(prev[path[-1]])
                return d, path[::-1]
            if d > dist[u]:
                continue
            for v, w in self.edges[u].items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        raise ValueError(f"no path from {a} to {b}")

    def shortest_path_m(self, a: str, b: str) -> float:
        return self.shortest_path(a, b)[0]

    def _add_node(self, name: str, xy: tuple[float, float]) -> None:
        self.nodes[name] = xy
        self.edges.setdefault(name, {})

    def _link(self, a: str, b: str) -> None:
        w = math.dist(self.nodes[a], self.nodes[b])
        self.edges[a][b] = w
        self.edges[b][a] = w


def load(path: str | Path = "store_layout") -> StoreMap:
    cfg = load_yaml(path)
    bay_w = float(cfg["bay"]["width_m"])
    per_face = int(cfg["bays_per_face"])
    run_depth = float(cfg["run_depth_m"])
    cap_w = float(cfg["end_cap"]["width_m"])
    cap_d = float(cfg["end_cap"]["depth_m"])
    aisle_w = float(cfg["aisles"]["width_m"])
    aisle_ids = list(cfg["aisles"]["ids"])
    front_y = float(cfg["cross_aisles"]["front"]["y_m"])
    back_y = float(cfg["cross_aisles"]["back"]["y_m"])

    runs = {
        r["id"]: Run(
            id=r["id"],
            x_m=float(r["x_m"]),
            y_m=float(r["y_m"]),
            length_m=float(r["length_m"]),
            depth_m=run_depth,
            tier=r["tier"],
            category=r.get("category", ""),
        )
        for r in cfg["runs"]
    }
    ordered = sorted(runs.values(), key=lambda r: r.x_m)
    if len(aisle_ids) != len(ordered) + 1:
        raise ValueError("need exactly one more aisle than runs")

    # Aisle k sits west of run k (k = 0..n-1); the last aisle is east of the last run.
    aisles = {}
    for k, aid in enumerate(aisle_ids):
        x = (
            ordered[k].x_m - aisle_w / 2
            if k < len(ordered)
            else ordered[-1].x_m + run_depth + aisle_w / 2
        )
        aisles[aid] = Aisle(id=aid, x_m=x, width_m=aisle_w)

    bays: dict[str, Bay] = {}
    for k, run in enumerate(ordered):
        for face, aid, x0, x1 in (
            ("L", aisle_ids[k], run.x_m, run.x_m + run_depth / 2),
            ("R", aisle_ids[k + 1], run.x_m + run_depth / 2, run.x_m + run_depth),
        ):
            for i in range(1, per_face + 1):
                y0 = run.y_m + (i - 1) * bay_w
                bid = f"{run.id}-{face}-{i:02d}"
                bays[bid] = Bay(
                    bay_id=bid,
                    run=run.id,
                    face=face,
                    index=i,
                    kind="bay",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y0 + bay_w,
                    aisle_id=aid,
                    front_xy=(aisles[aid].x_m, y0 + bay_w / 2),
                    tier=run.tier,
                )
        cx = run.x_m + run_depth / 2
        y_end = run.y_m + run.length_m
        for end, walkway, y0, y1, wy in (
            ("F", "front", run.y_m - cap_d, run.y_m, front_y),
            ("B", "back", y_end, y_end + cap_d, back_y),
        ):
            bid = f"{run.id}-E-{end}"
            bays[bid] = Bay(
                bay_id=bid,
                run=run.id,
                face="E",
                index=None,
                kind="end_cap",
                x0=cx - cap_w / 2,
                x1=cx + cap_w / 2,
                y0=y0,
                y1=y1,
                aisle_id=walkway,
                front_xy=(cx, wy),
                tier=run.tier,
            )

    sm = StoreMap(
        runs=runs,
        aisles=aisles,
        bays=bays,
        cross_aisles=dict(cfg["cross_aisles"]),
        dock_xy=tuple(cfg["backroom"]["dock"]),
    )
    _build_graph(sm, aisle_ids, ordered, per_face, front_y, back_y, tuple(cfg["backroom"]["door"]))
    return sm


def _build_graph(sm, aisle_ids, ordered, per_face, front_y, back_y, door_xy) -> None:
    """Walkways: each aisle is a chain front end -> bay stops -> back end; the front and back
    walkways chain aisle ends and end-cap stops west to east; the dock joins via the door."""
    for aid in aisle_ids:
        x = sm.aisles[aid].x_m
        chain = [f"{aid}:F"]
        sm._add_node(chain[0], (x, front_y))
        for i in range(1, per_face + 1):
            stop = f"{aid}:{i:02d}"
            y = ordered[0].y_m + (i - 0.5) * (ordered[0].length_m / per_face)
            sm._add_node(stop, (x, y))
            chain.append(stop)
        chain.append(f"{aid}:B")
        sm._add_node(chain[-1], (x, back_y))
        for a, b in zip(chain, chain[1:], strict=False):
            sm._link(a, b)

    for end in ("F", "B"):
        stops = [f"{aid}:{end}" for aid in aisle_ids]
        for run in ordered:
            cap = sm.bays[f"{run.id}-E-{end}"]
            stop = f"{run.id}:{end}"
            sm._add_node(stop, cap.front_xy)
            stops.append(stop)
        stops.sort(key=lambda n: sm.nodes[n][0])
        for a, b in zip(stops, stops[1:], strict=False):
            sm._link(a, b)

    # Bays are leaf nodes at their walkway stop (zero-length edge).
    for b in sm.bays.values():
        if b.kind == "bay":
            stop = f"{b.aisle_id}:{b.index:02d}"
        else:
            stop = f"{b.run}:{'F' if b.aisle_id == 'front' else 'B'}"
        sm._add_node(b.bay_id, sm.nodes[stop])
        sm._link(b.bay_id, stop)

    sm._add_node(DOOR, door_xy)
    sm._add_node(DOCK, sm.dock_xy)
    east_front = max((f"{aid}:F" for aid in aisle_ids), key=lambda n: sm.nodes[n][0])
    sm._link(DOOR, east_front)
    sm._link(DOCK, DOOR)
