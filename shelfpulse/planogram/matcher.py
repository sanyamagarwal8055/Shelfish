"""Compare one BayReading with the bay's planogram: a status per slot, plus stray packs.

Positions come from rectified images in cm, so each pack is placed by geometry: it belongs to
the slot that contains its centre. Per slot:

- facings = packs of the planned SKU, counting an `AMBIGUOUS:` pack if the planned SKU is one
  of its candidates (so ambiguity never causes a false OUT);
- OUT if none, LOW if fewer than min_facings, else OK;
- an OUT/LOW slot becomes UNKNOWN ("can't tell") when it overlaps an occluded range (never
  alert on what is hidden) or holds UNKNOWN packs (packs are there but Vision couldn't name
  them, e.g. without its gallery index), and so does every slot of a row the reading omits.

Any other pack is a stray: MISPLACED (a real, different SKU), UNKNOWN_ITEM (not in the gallery)
or AMBIGUOUS (candidates don't include the planned SKU; never reported as MISPLACED).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shelfpulse.contracts import (
    UNKNOWN_SKU,
    BayReading,
    Pack,
    Planogram,
    PlanogramSlot,
    Row,
    ambiguous_candidates,
    is_ambiguous,
)
from shelfpulse.decision.types import SlotObservation, StrayItem, StrayKind


@dataclass
class MatchResult:
    slots: list[SlotObservation] = field(default_factory=list)
    strays: list[StrayItem] = field(default_factory=list)


def match(plan: Planogram, reading: BayReading) -> MatchResult:
    if plan.bay_id != reading.bay_id:
        raise ValueError(f"planogram {plan.bay_id} does not match reading {reading.bay_id}")
    seen = {r.row: r for r in reading.rows}
    out = MatchResult()
    for row_no in sorted(set(range(len(plan.rows))) | set(seen)):
        slots = plan.rows[row_no] if row_no < len(plan.rows) else []
        _match_row(row_no, slots, seen.get(row_no), reading, out)
    return out


def _match_row(
    row_no: int, slots: list[PlanogramSlot], row: Row | None, reading: BayReading, out: MatchResult
) -> None:
    if row is None:  # row not seen
        out.slots += [_obs(row_no, s, [], "UNKNOWN", reading) for s in slots]
        return

    by_slot: dict[int, list[Pack]] = {i: [] for i in range(len(slots))}
    for p in row.packs:
        i = _slot_of(p, slots)
        if i is None:
            out.strays.append(_stray(row_no, None, None, p, reading))
        else:
            by_slot[i].append(p)

    for i, s in enumerate(slots):
        counted = []
        for p in by_slot[i]:
            if s.sku_id in ambiguous_candidates(p.sku):
                counted.append(p)
            else:
                out.strays.append(_stray(row_no, s.position, s.sku_id, p, reading))
        status = _status(len(counted), s)
        unnamed = any(p.sku == UNKNOWN_SKU for p in by_slot[i])
        if status != "OK" and (unnamed or _overlaps(s, row.occluded)):
            status = "UNKNOWN"
        out.slots.append(_obs(row_no, s, counted, status, reading))


def _slot_of(p: Pack, slots: list[PlanogramSlot]) -> int | None:
    centre = p.x_cm + p.w_cm / 2
    for i, s in enumerate(slots):
        last = i == len(slots) - 1
        if s.x_start_cm <= centre < s.x_end_cm or (last and centre == s.x_end_cm):
            return i
    return None


def _status(facings: int, s: PlanogramSlot) -> str:
    if facings == 0 and s.facings > 0:
        return "OUT"
    if facings < s.min_facings:
        return "LOW"
    return "OK"


def _overlaps(s: PlanogramSlot, occluded: list[list[float]]) -> bool:
    return any(a < s.x_end_cm and b > s.x_start_cm for a, b in occluded)


def _obs(
    row_no: int, s: PlanogramSlot, counted: list[Pack], status: str, reading: BayReading
) -> SlotObservation:
    if status == "UNKNOWN":
        units = None
    elif any(p.depth_left is None for p in counted):
        units = None
    else:
        units = sum(p.stack * p.depth_left for p in counted)  # 0 when OUT: empty is empty
    return SlotObservation(
        bay_id=reading.bay_id,
        row=row_no,
        position=s.position,
        sku=s.sku_id,
        status=status,
        facings=len(counted),
        planned_facings=s.facings,
        min_facings=s.min_facings,
        units=units,
        ambiguous=any(is_ambiguous(p.sku) for p in counted),
        conf=min((p.conf for p in counted), default=None),
        source=reading.source,
        t=reading.t,
    )


def _stray(
    row_no: int, position: int | None, expected: str | None, p: Pack, reading: BayReading
) -> StrayItem:
    kind: StrayKind
    if p.sku == UNKNOWN_SKU:
        kind = "UNKNOWN_ITEM"
    elif is_ambiguous(p.sku):
        kind = "AMBIGUOUS"
    else:
        kind = "MISPLACED"
    return StrayItem(
        kind=kind,
        bay_id=reading.bay_id,
        row=row_no,
        position=position,
        expected=expected,
        sku=p.sku,
        x_cm=p.x_cm,
        conf=p.conf,
        source=reading.source,
        t=reading.t,
    )
