"""Sales-based stock estimate for shelves whose depth the cameras can't see.

The camera always wins what it can see: an empty slot is empty and a depth count is the count
(both handled by the matcher). Only when a slot looks fine but its depth is unknown does the
estimate matter: packs on the shelves = the store system's on-hand minus the backroom (its own
"last count minus sales since"). If that is below what the next restock lead time will sell,
the slot is LOW_ESTIMATED: front faced up, little behind. That is never a staff alert on its
own; the robot is sent to measure depth and settle it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from shelfpulse.decision.store_data import StoreData


class QuantityCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    restock_lead_h: float = Field(gt=0)  # time staff need to refill once told
    safety_units: int = Field(ge=0)


def shelf_estimate(sku: str, t: datetime, data: StoreData) -> int | None:
    """Packs of the SKU on the shelves according to the store system, or None if unknown."""
    on_hand, backroom = data.system_on_hand(sku, t), data.backroom(sku, t)
    if on_hand is None or backroom is None:
        return None
    return max(0, on_hand - backroom)


def low_estimated(estimate: int | None, velocity: float, cfg: QuantityCfg) -> bool:
    """True if the estimate won't last the restock lead time (plus a safety margin)."""
    if estimate is None:
        return False
    return estimate < velocity * cfg.restock_lead_h + cfg.safety_units
