"""Why is a slot empty or low? Checked in a fixed order, theft first.

1. THEFT          packs dropped by >= theft_min_units within theft_window_min before the alert,
                  while POS sold at most theft_max_sales_frac of that drop
2. REPLENISHMENT  the backroom has the SKU: staff can refill now
3. PHANTOM_STOCK  backroom empty, but the system on-hand exceeds what is on the shelves by
                  >= phantom_min_units: the count is wrong
4. STOCKOUT       nothing anywhere: someone has to re-order

Without POS or stock data only the visible fact is known (NO_STORE_DATA).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shelfpulse.decision.store_data import StoreData

Cause = Literal["THEFT", "REPLENISHMENT_GAP", "PHANTOM_STOCK", "TRUE_STOCKOUT", "NO_STORE_DATA"]


class DiagnosisCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    theft_window_min: float = Field(gt=0)
    theft_min_units: int = Field(ge=1)
    theft_max_sales_frac: float = Field(ge=0.0, le=1.0)
    phantom_min_units: int = Field(ge=1)


@dataclass(frozen=True)
class Diagnosis:
    cause: Cause
    theft_units: int = 0  # packs that vanished without a sale (THEFT only)
    backroom: int | None = None
    phantom_units: int = 0  # system on-hand not on any shelf or in the backroom


def diagnose(
    sku: str,
    since: datetime,
    t: datetime,
    history: list[tuple[datetime, int]],
    shelf_units: int,
    data: StoreData | None,
    cfg: DiagnosisCfg,
) -> Diagnosis:
    """`history`: (time, packs seen) for the slot; `shelf_units`: packs of the SKU seen on
    every shelf in the store right now."""
    if data is None:
        return Diagnosis("NO_STORE_DATA")

    window_start = since - timedelta(minutes=cfg.theft_window_min)
    before = [u for ts, u in history if window_start <= ts < since]
    now = history[-1][1] if history else 0
    drop = max(before, default=now) - now
    if drop >= cfg.theft_min_units:
        sold = data.sold(sku, window_start, t)
        if sold <= drop * cfg.theft_max_sales_frac:
            return Diagnosis("THEFT", theft_units=drop - sold, backroom=data.backroom(sku, t))

    backroom = data.backroom(sku, t)
    if backroom is None:
        return Diagnosis("NO_STORE_DATA")
    if backroom > 0:
        return Diagnosis("REPLENISHMENT_GAP", backroom=backroom)
    on_hand = data.system_on_hand(sku, t) or 0
    phantom = on_hand - backroom - shelf_units
    if phantom >= cfg.phantom_min_units:
        return Diagnosis("PHANTOM_STOCK", backroom=0, phantom_units=phantom)
    return Diagnosis("TRUE_STOCKOUT", backroom=0)
