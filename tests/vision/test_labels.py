"""Step 7b: shelf-edge price labels read from robot images."""

from __future__ import annotations

import pytest

from shelfpulse.contracts import load_sku_master
from shelfpulse.perception.labels import LabelReader

SKUS = load_sku_master()
READER = LabelReader(SKUS)


def test_sku_from_exact_id_or_product_name():
    assert READER.sku_of("RICE_5KG") == "RICE_5KG"
    assert READER.sku_of("Toothpaste 150 g") == "TOOTHPASTE_150G"  # "150 g" matches "150g"
    assert READER.sku_of("Toor Dal 1 kg") == "DAL_1KG"  # 3 keywords beat DAL_500G's 2
    assert READER.sku_of("Basmati Rice 5 kg") == "RICE_5KG"


def test_unclear_text_names_no_sku():
    assert READER.sku_of("Toor Dal") is None  # DAL_1KG and DAL_500G tie: never guess
    assert READER.sku_of("Special offer") is None


@pytest.mark.parametrize(
    ("text", "price"),
    [("Rs 89.00", 89.0), ("Rs89", 89.0), ("₹ 30,50", 30.5), ("MRP 145.5", 145.5), ("none", None)],
)
def test_price_parsing(text, price):
    assert LabelReader.price_of(text) == price


def test_robot_readings_carry_the_labels(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    from shelfpulse.contracts import parse_mission
    from shelfpulse.perception.analyze import load_pipeline
    from shelfpulse.sources.robot import bridge
    from tools.eval_readings import evaluate, load
    from tools.synth.robot import record

    bays = ["G3-L-05", "G7-R-02", "G9-E-F"]  # G9-E-F: random planogram, non-food names
    record(tmp_path / "rec", bays, seed=2)
    m = parse_mission(
        {"contract_version": "1.0", "mission_id": "M-0003", "kind": "mission", "bays": bays,
         "created_at": "2026-10-08T11:00:00+05:30", "speed_mps": 0.3}
    )  # fmt: skip
    b = bridge.run([m], tmp_path / "out", tmp_path / "rec", pipeline=load_pipeline(identify=False))
    assert len(b.readings) == 3
    pred = load(tmp_path / "out" / "bay_readings.jsonl")
    o = evaluate(pred, load(tmp_path / "rec" / "truth.jsonl"), by_bay=True).overall
    assert o.labels > 20
    assert o.labels_found == o.labels == o.labels_pred  # every tag read, nothing invented
    assert o.labels_price_ok == o.labels_found
