"""Step 2: the matcher turns BayReading rows into slot statuses and stray packs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shelfpulse.contracts import parse_bay_reading, parse_planogram
from shelfpulse.planogram.loader import PlanogramStore
from shelfpulse.planogram.matcher import match

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "contracts" / "fixtures" / "demo_store" / "planograms"

# G1-L-04 row 3: RICE_1KG 0-60 cm (6 facings, min 2), DAL_1KG 60-120 cm (6 facings, min 2).
PLAN = parse_planogram(json.loads((DEMO / "G1-L-04.json").read_text()))


def pack(sku, x, depth=3, stack=1, conf=0.9, w=9.5):
    return {"sku": sku, "conf": conf, "x_cm": x, "w_cm": w, "h_cm": 25.0, "stack": stack,
            "depth_left": depth}  # fmt: skip


def reading(packs, occluded=(), row=3, source="camera"):
    return parse_bay_reading(
        {
            "bay_id": "G1-L-04",
            "source": source,
            "t": "2026-10-08T11:20:00+05:30",
            "frame_ref": "f.jpg",
            "quality": 0.9,
            "px_per_cm": 10.0,
            "rows": [{"row": row, "occluded": [list(o) for o in occluded], "packs": packs}],
        }
    )


def full_row():
    return [pack("RICE_1KG", 10.0 * i) for i in range(6)] + [
        pack("DAL_1KG", 60 + 10.0 * i) for i in range(6)
    ]


def row3(result):
    return {o.sku: o for o in result.slots if o.row == 3}


def test_full_row_is_ok():
    s = row3(match(PLAN, reading(full_row())))
    assert s["RICE_1KG"].status == "OK" and s["RICE_1KG"].facings == 6
    assert s["DAL_1KG"].status == "OK"
    assert s["RICE_1KG"].units == 18  # 6 facings x depth 3


@pytest.mark.parametrize("n, status", [(6, "OK"), (2, "OK"), (1, "LOW"), (0, "OUT")])
def test_low_and_out(n, status):
    packs = [pack("RICE_1KG", 10.0 * i) for i in range(n)] + full_row()[6:]
    assert row3(match(PLAN, reading(packs)))["RICE_1KG"].status == status


def test_out_has_zero_units():
    s = row3(match(PLAN, reading(full_row()[6:])))["RICE_1KG"]
    assert (s.status, s.units) == ("OUT", 0)


def test_unknown_depth_gives_no_units():
    packs = full_row()
    packs[0]["depth_left"] = None
    assert row3(match(PLAN, reading(packs)))["RICE_1KG"].units is None


def test_misplaced_pack_is_a_stray_not_a_facing():
    packs = full_row()
    packs[1] = pack("SHAMPOO_180ML", 10.0)
    m = match(PLAN, reading(packs))
    assert row3(m)["RICE_1KG"].facings == 5
    (st,) = m.strays
    assert (st.kind, st.sku, st.expected, st.position) == (
        "MISPLACED", "SHAMPOO_180ML", "RICE_1KG", 0,
    )  # fmt: skip


def test_unknown_pack_is_unknown_item():
    packs = full_row()
    packs[0] = pack("UNKNOWN", 0.0)
    (st,) = match(PLAN, reading(packs)).strays
    assert st.kind == "UNKNOWN_ITEM"


def test_ambiguous_with_planned_candidate_counts_as_facing():
    packs = full_row()
    packs[6] = pack("AMBIGUOUS:DAL_1KG|DAL_500G", 60.0, depth=None)
    m = match(PLAN, reading(packs))
    dal = row3(m)["DAL_1KG"]
    assert dal.facings == 6 and dal.ambiguous and dal.units is None
    assert m.strays == []


def test_ambiguous_without_planned_candidate_is_never_misplaced():
    packs = full_row()
    packs[0] = pack("AMBIGUOUS:SOAP_100G|SHAMPOO_180ML", 0.0)
    (st,) = match(PLAN, reading(packs)).strays
    assert st.kind == "AMBIGUOUS"


def test_pack_belongs_to_slot_holding_its_centre():
    # x 55 + w 9.5 -> centre 59.75, inside RICE (0-60) though it crosses the boundary.
    packs = [pack("RICE_1KG", 55.0), pack("RICE_1KG", 45.0)] + full_row()[6:]
    m = match(PLAN, reading(packs))
    assert row3(m)["RICE_1KG"].facings == 2 and m.strays == []


def test_occluded_empty_slot_is_unknown_not_out():
    s = row3(match(PLAN, reading(full_row()[:6], occluded=[(80.0, 120.0)])))
    assert s["DAL_1KG"].status == "UNKNOWN" and s["DAL_1KG"].units is None
    assert s["RICE_1KG"].status == "OK"


def test_occluded_but_visibly_full_slot_stays_ok():
    s = row3(match(PLAN, reading(full_row(), occluded=[(100.0, 120.0)])))
    assert s["DAL_1KG"].status == "OK"


def test_rows_not_in_reading_are_unknown():
    m = match(PLAN, reading(full_row()))
    others = [o for o in m.slots if o.row != 3]
    assert others and all(o.status == "UNKNOWN" for o in others)
    assert len(m.slots) == sum(len(r) for r in PLAN.rows)


def test_seen_empty_row_is_out():
    m = match(PLAN, reading([], row=2))  # row 2: RICE_1KG across the whole bay
    (o,) = [o for o in m.slots if o.row == 2]
    assert o.status == "OUT"


def test_wrong_bay_rejected():
    r = reading(full_row()).model_copy(update={"bay_id": "G1-L-05"})
    with pytest.raises(ValueError):
        match(PLAN, r)


def test_store_search_order(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    lm = {**json.loads((DEMO / "G1-L-04.json").read_text()), "source": "label_map"}
    (second / "G1-L-04.json").write_text(json.dumps(lm))
    store = PlanogramStore([first, second, DEMO])
    assert store.get("G1-L-04").source == "label_map"  # b beats DEMO
    assert store.get("G1-L-05").source == "planogram"  # only in DEMO
    assert store.get("G9-L-01") is None
