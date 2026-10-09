"""Step 5: naming packs. Decision rules need no model; one test runs the real DINOv2 if present."""

from __future__ import annotations

import numpy as np
import pytest

from shelfpulse.contracts import UNKNOWN_SKU, load_sku_master
from shelfpulse.perception.identify import Identifier, IdentifySettings, PlanHints

SKUS = load_sku_master()
DEMO = ["contracts/fixtures/demo_store/planograms"]
S = IdentifySettings(unknown_below=0.5, ambiguous_margin=0.02, location_bonus=0.05, size_bonus=0.05)


def _ident(hints: bool = True) -> Identifier:
    return Identifier(None, None, SKUS, PlanHints(DEMO) if hints else None, S)  # decide() only


def test_plan_hints_read_the_contract_planogram():
    h = PlanHints(DEMO)
    assert h.planned("G1-L-04", 0, 10.0) == "RICE_1KG"
    assert h.planned("G1-L-04", 0, 75.0) == "RICE_5KG"
    assert h.planned("G9-L-01", 0, 10.0) is None  # no planogram for that bay


def test_clear_winner():
    sku, conf = _ident().decide([("DAL_1KG", 0.80), ("DAL_500G", 0.60)], "G1-L-04", 1, 0.0, 9.0)
    assert (sku, conf) == ("DAL_1KG", 0.8)


def test_below_threshold_is_unknown_even_where_planned():
    sku, _ = _ident().decide([("RICE_1KG", 0.40)], "G1-L-04", 0, 0.0, 9.5)
    assert sku == UNKNOWN_SKU


def test_close_call_is_ambiguous():
    # Same width (9 cm) and neither planned at x=200 -> only similarity decides; too close to call.
    sku, _ = _ident(hints=False).decide(
        [("DAL_1KG", 0.700), ("DAL_500G", 0.695)], "G1-L-04", 1, 0.0, 9.0
    )
    assert sku == "AMBIGUOUS:DAL_1KG|DAL_500G"


def test_location_breaks_a_tie():
    # x = 0-9 cm on row 1 of G1-L-04 is planned DAL_1KG.
    sku, _ = _ident().decide([("DAL_500G", 0.700), ("DAL_1KG", 0.695)], "G1-L-04", 1, 0.0, 9.0)
    assert sku == "DAL_1KG"


def test_location_does_not_hide_a_clearly_misplaced_pack():
    # Shampoo standing in the rice slot: the planned SKU's bonus must not outvote a clear match.
    sku, _ = _ident().decide([("SHAMPOO_180ML", 0.85), ("RICE_1KG", 0.62)], "G1-L-04", 0, 0.0, 6.0)
    assert sku == "SHAMPOO_180ML"


def test_size_breaks_a_tie():
    # 30 cm wide: RICE_5KG (30 cm) fits, RICE_1KG (9.5 cm) doesn't. Hints off.
    sku, _ = _ident(hints=False).decide(
        [("RICE_1KG", 0.700), ("RICE_5KG", 0.690)], "G1-L-04", 0, 30.0, 30.0
    )
    assert sku == "RICE_5KG"


def test_index_search_and_round_trip(tmp_path):
    faiss = pytest.importorskip("faiss")
    from shelfpulse.perception.sku_index import SkuIndex

    vecs = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vecs)
    idx = SkuIndex(index, ["A", "A", "B"], "fake@3")
    idx.save(tmp_path / "idx")
    back = SkuIndex.load(tmp_path / "idx")
    hits = back.search(np.array([[0.0, 0.6, 0.8]], np.float32), k=2)[0]
    assert hits == [("B", pytest.approx(0.8)), ("A", pytest.approx(0.6))]  # one entry per SKU
    assert back.model == "fake@3"


def test_real_embedder_prefers_the_same_product(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("faiss")
    import cv2

    from shelfpulse.perception.embedder import Embedder
    from shelfpulse.perception.sku_index import SkuIndex

    try:
        emb = Embedder("dinov2")
    except OSError as e:  # model not downloaded and no internet
        pytest.skip(f"DINOv2 not available: {e}")
    rng = np.random.default_rng(0)
    photos = {}
    for sku in ("A", "B"):
        d = tmp_path / sku
        d.mkdir()
        img = rng.integers(0, 255, (8, 6, 3), dtype=np.uint8)  # distinct coarse patterns
        img = cv2.resize(img, (120, 160), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(d / "a.jpg"), img)
        photos[sku] = [d / "a.jpg"]
    index = SkuIndex.build(photos, emb)
    query = cv2.imread(str(tmp_path / "B" / "a.jpg"))[10:-10, 5:-5]  # cropped copy of B
    assert index.search(emb.embed([query]), k=2)[0][0][0] == "B"
