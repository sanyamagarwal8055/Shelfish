"""Load YAML settings from configs/ (shared). Every tunable number lives in configs/*.yaml."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

HHMM = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
HHMM_RANGE = Annotated[
    str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$")
]


def config_path(name_or_path: str | Path) -> Path:
    """`"robot"` -> configs/robot.yaml; an existing path or a *.yaml name is used as given."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() or p.exists() else REPO_ROOT / p
    return CONFIG_DIR / f"{name_or_path}.yaml"


def load_yaml(name_or_path: str | Path) -> dict:
    with open(config_path(name_or_path), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MissionLimits(_Cfg):
    max_bays: int = Field(ge=1)
    max_per_hour: int = Field(ge=0)
    blackout: list[HHMM_RANGE] = Field(default_factory=list)

    def blackout_windows(self) -> list[tuple[time, time]]:
        return [(_hhmm(a), _hhmm(b)) for a, b in (w.split("-") for w in self.blackout)]


class PeopleRules(_Cfg):
    max_person_cover: float = Field(ge=0.0, le=1.0)


class Triggers(_Cfg):
    camera_blocked_min: float = Field(gt=0)


class RobotConfig(_Cfg):
    sweeps: list[HHMM]
    speed_sweep_mps: float = Field(gt=0)
    speed_mission_mps: float = Field(gt=0)
    missions: MissionLimits
    people: PeopleRules
    triggers: Triggers

    def sweep_times(self) -> list[time]:
        return [_hhmm(s) for s in self.sweeps]


def load_robot_config(name_or_path: str | Path = "robot") -> RobotConfig:
    return RobotConfig.model_validate(load_yaml(name_or_path))
