"""The robot vendor's API, and a fake one that replays synthetic recordings.

A real vendor API takes waypoints and streams back frames with poses and status. FakeVendorAPI
does the same for the demo: it drives the given bays in order on the store's walkway graph
(travel time = walking distance / speed), and for each bay replays the frames of a recording
(tools/synth/robot.py: pose.csv + frames + depth) whose pose shows that bay. Bays the recording
lacks are rendered on the fly with tools.synth.robot, so any mission works without a recording.
A real walk-along video + pose.csv can be added as another recording reader.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from shelfpulse.config import REPO_ROOT
from shelfpulse.layout.store_map import DOCK, StoreMap
from shelfpulse.sources.robot.geometry import BayLocator, Pose


@dataclass
class Frame:
    image: np.ndarray
    depth: np.ndarray
    pose: Pose


@dataclass
class Event:
    kind: str  # "arrived" (at a bay), "frame", "passed" (bay pass done), "docked"
    t: datetime
    xy: tuple[float, float]
    bay_id: str | None = None
    frame: Frame | None = None


def _read(path: Path, flags: int) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


@dataclass
class Recording:
    """A robot recording on disk: pose.csv rows, frames loaded on demand."""

    root: Path
    rows: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> Recording:
        with open(root / "pose.csv", newline="", encoding="utf-8") as f:
            return cls(root, list(csv.DictReader(f)))

    def frames_for(self, bay_id: str, locator: BayLocator, width_cm: float) -> list[Frame]:
        out = []
        for r in self.rows:
            pose = Pose(datetime.fromisoformat(r["t"]), float(r["x_m"]), float(r["y_m"]),
                        float(r["heading_deg"]), r["side"])  # fmt: skip
            if any(b == bay_id for b, _ in locator.in_view(pose, width_cm)):
                image = _read(self.root / r["frame_file"], cv2.IMREAD_COLOR)
                depth = _read(self.root / r["depth_file"], cv2.IMREAD_UNCHANGED)
                out.append(Frame(image, depth, pose))
        return out


class FakeVendorAPI:
    def __init__(
        self,
        smap: StoreMap,
        recording: Path | None = None,
        *,
        render_missing: bool = True,
        people: set[str] = frozenset(),
        seed: int = 0,
        frame_width_cm: float = 60.0,
    ):
        self.smap = smap
        self.locator = BayLocator(smap)
        self.recording = Recording.load(recording) if recording else None
        self.render_missing = render_missing
        self.people = set(people)
        self.rng = np.random.default_rng(seed)
        self.frame_width_cm = frame_width_cm
        self.node = DOCK
        self.xy = smap.dock_xy
        self.t: datetime | None = None
        self._route: list[str] = []
        self._speed = 0.3
        self._synth = None  # (skus, gallery) for on-the-fly rendering

    # --- the vendor API surface ------------------------------------------------------------

    def send_waypoints(self, bays: list[str], speed_mps: float, start: datetime) -> None:
        unknown = [b for b in bays if b not in self.smap.bays]
        if unknown:
            raise ValueError(f"unknown bays {unknown}")
        self._route, self._speed = list(bays), speed_mps
        self.t = start if self.t is None else max(self.t, start)

    def frames(self) -> Iterator[Event]:
        """Drive the route: per bay, an 'arrived' event, its frames, then 'passed'."""
        for bay_id in self._route:
            self._travel(bay_id)
            yield Event("arrived", self.t, self.xy, bay_id)
            frames = self._frames(bay_id)
            if frames:
                t0, prev = self.t, None
                for f in frames:
                    if prev is not None:
                        step = math.dist((prev.x_m, prev.y_m), (f.pose.x_m, f.pose.y_m))
                        self.t += timedelta(seconds=step / self._speed)
                    prev = f.pose
                    pose = Pose(self.t, f.pose.x_m, f.pose.y_m, f.pose.heading_deg, f.pose.side)
                    self.xy = (pose.x_m, pose.y_m)
                    yield Event("frame", self.t, self.xy, bay_id, Frame(f.image, f.depth, pose))
                if self.t == t0:
                    self.t += timedelta(seconds=1)
            yield Event("passed", self.t, self.xy, bay_id)
        self._route = []

    def dock(self) -> Event:
        self._travel(DOCK)
        return Event("docked", self.t, self.xy)

    def status(self) -> tuple[datetime | None, tuple[float, float]]:
        return self.t, self.xy

    # --- internals -------------------------------------------------------------------------

    def _travel(self, node: str) -> None:
        dist = self.smap.shortest_path_m(self.node, node)
        self.t += timedelta(seconds=dist / self._speed)
        self.node = node
        self.xy = self.smap.nodes[node] if node == DOCK else self.smap.bays[node].front_xy

    def _frames(self, bay_id: str) -> list[Frame]:
        if self.recording is not None:
            frames = self.recording.frames_for(bay_id, self.locator, self.frame_width_cm)
            if frames:
                return frames
        if not self.render_missing:
            return []
        return self._render(bay_id)

    def _render(self, bay_id: str) -> list[Frame]:
        from shelfpulse.contracts import load_sku_master  # synthetic world only
        from tools.synth.make import load_gallery, load_geometry, planograms_for
        from tools.synth.robot import render_pass

        if self._synth is None:
            self._synth = (load_sku_master(), load_gallery(REPO_ROOT / "data" / "gallery"))
        skus, gallery = self._synth
        plano = planograms_for([bay_id], None, skus, load_geometry(), self.rng)[0]
        frames, _ = render_pass(bay_id, plano, skus, self.rng, self.t, gallery=gallery,
                                person=bay_id in self.people, speed_mps=self._speed)  # fmt: skip
        return [Frame(f.image, f.depth, f.pose) for f in frames]
