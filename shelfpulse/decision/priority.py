"""Rupees lost per hour, and the P1/P2/P3 bucket staff see.

An OUT slot loses its whole sales rate: velocity (packs/h) x margin (Rs per pack). A LOW slot
is still selling, so it counts at `low_weight` of that. Some actions have a floor priority
whatever the money (a theft note is always P1; a wrong stock count blocks re-ordering).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PRIORITIES = ("P1", "P2", "P3")


class PriorityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    velocity_window_h: float = Field(gt=0)
    min_velocity_span_h: float = Field(gt=0)
    low_weight: float = Field(ge=0.0, le=1.0)
    p1_min_rupees_per_h: float = Field(ge=0)
    p2_min_rupees_per_h: float = Field(ge=0)
    min_priority: dict[str, Literal["P1", "P2", "P3"]] = Field(default_factory=dict)


def rupees_per_h(kind: str, velocity: float, margin_inr: float, cfg: PriorityCfg) -> float:
    weight = cfg.low_weight if kind in ("LOW", "LOW_ESTIMATED") else 1.0
    return round(velocity * margin_inr * weight, 2)


def bucket(rupees: float, action: str, cfg: PriorityCfg) -> str:
    if rupees >= cfg.p1_min_rupees_per_h:
        p = "P1"
    elif rupees >= cfg.p2_min_rupees_per_h:
        p = "P2"
    else:
        p = "P3"
    floor = cfg.min_priority.get(action)
    if floor is not None and PRIORITIES.index(floor) < PRIORITIES.index(p):
        p = floor
    return p
