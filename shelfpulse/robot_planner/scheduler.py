"""When the robot goes: the two daily sweeps, and on-demand missions when allowed.

Sweeps (robot.yaml `sweeps`) go out on the first tick at or after the sweep time, unless the
clock is already more than `sweep_grace_min` late. A mission goes out when the queue has a due
bay, the robot is free, and the store allows it:

- not inside a blackout window (robot.yaml `missions.blackout`, peak hours),
- fewer than `missions.max_per_hour` missions in the last hour,
- POS transactions in the last 10 minutes at most `footfall_tx_per_10min_max` (busy aisles).

A mission takes up to `missions.max_bays` bays, ordered by `route.order`. The robot counts as
busy for the walk time (dock -> bays -> dock at mission speed, plus `bay_scan_s` per bay) or
the sweep duration, unless a RobotStatus says it is docked or idle sooner.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from shelfpulse.config import RobotConfig
from shelfpulse.contracts import Mission, RobotStatus
from shelfpulse.decision.store_data import StoreData
from shelfpulse.layout.store_map import StoreMap
from shelfpulse.robot_planner import route
from shelfpulse.robot_planner.mission_queue import MissionQueue, PlannerCfg


class Scheduler:
    def __init__(
        self,
        robot: RobotConfig,
        planner: PlannerCfg,
        footfall_max: int,
        smap: StoreMap,
        queue: MissionQueue,
        store_data: StoreData | None = None,
    ):
        self.robot, self.cfg, self.footfall_max = robot, planner, footfall_max
        self.smap, self.queue, self.store_data = smap, queue, store_data
        self.busy_until: datetime | None = None
        self.sent: list[Mission] = []
        self._swept: set[tuple] = set()
        self._next_id = 1

    # --- gating -----------------------------------------------------------------------------------

    def in_blackout(self, now: datetime) -> bool:
        clock = now.timetz().replace(tzinfo=None)
        return any(a <= clock < b for a, b in self.robot.missions.blackout_windows())

    def busy_store(self, now: datetime) -> bool:
        if self.store_data is None:
            return False
        return self.store_data.transactions(now - timedelta(minutes=10), now) > self.footfall_max

    def missions_last_hour(self, now: datetime) -> int:
        hour_ago = now - timedelta(hours=1)
        return sum(1 for m in self.sent if m.kind == "mission" and hour_ago < m.created_at <= now)

    def allowed(self, now: datetime) -> bool:
        return (
            not self.in_blackout(now)
            and self.missions_last_hour(now) < self.robot.missions.max_per_hour
            and not self.busy_store(now)
        )

    def free(self, now: datetime) -> bool:
        return self.busy_until is None or now >= self.busy_until

    # --- robot feedback ---------------------------------------------------------------------------

    def on_status(self, st: RobotStatus) -> None:
        """RobotStatus from the bridge: docked/idle frees the robot; skipped bays re-queue."""
        if st.state in ("docked", "idle"):
            self.busy_until = st.t
        for bay in st.skipped_bays:
            self.queue.add(bay, "blocked", st.t)

    # --- tick -------------------------------------------------------------------------------------

    def tick(self, now: datetime) -> list[Mission]:
        out = []
        sweep = self._sweep_due(now)
        if sweep is not None and self.free(now):
            self._swept.add((now.date(), sweep))
            out.append(self._mission("sweep", [], {}, self.robot.speed_sweep_mps, now))
            self.busy_until = now + timedelta(minutes=self.cfg.sweep_duration_min)
        elif self.free(now) and self.queue.ready(now) and self.allowed(now):
            entries = self.queue.pop(self.robot.missions.max_bays, now)
            bays = route.order([e.bay_id for e in entries], self.smap)
            reasons = {e.bay_id: e.reason for e in entries}
            speed = self.robot.speed_mission_mps
            out.append(self._mission("mission", bays, reasons, speed, now))
            walk_s = route.walk_m(bays, self.smap) / speed + self.cfg.bay_scan_s * len(bays)
            self.busy_until = now + timedelta(seconds=walk_s)
        self.sent += out
        return out

    def _sweep_due(self, now: datetime) -> time | None:
        clock = now.timetz().replace(tzinfo=None)
        for s in self.robot.sweep_times():
            start = datetime.combine(now.date(), s, now.tzinfo)
            late = now - start
            if timedelta(0) <= late <= timedelta(minutes=self.cfg.sweep_grace_min):
                if (now.date(), s) not in self._swept and clock >= s:
                    return s
        return None

    def _mission(self, kind: str, bays: list[str], reasons: dict[str, str], speed: float,
                 now: datetime) -> Mission:  # fmt: skip
        m = Mission(mission_id=f"M-{self._next_id:04d}", kind=kind, created_at=now, bays=bays,
                    reasons=reasons, speed_mps=speed)  # fmt: skip
        self._next_id += 1
        return m
