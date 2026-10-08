"""Step 2: tools/eval_readings.py scores truth against itself as perfect and catches mistakes."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tools.eval_readings import evaluate, frame_key, load
from tools.synth.make import make


@pytest.fixture(scope="module")
def gold(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval_gold")
    make(out, n=4, seed=2, gaps=True)
    return out / "truth.jsonl"


def test_truth_vs_itself_is_perfect(gold):
    o = evaluate(load(gold), load(gold)).overall
    assert o.facing_err == 0.0 and o.sku_acc == 1.0 and o.gap_recall == 1.0
    assert o.packs > 0 and o.gaps > 0


def test_dropped_pack_wrong_sku_missed_gap_lower_scores(gold):
    pred = load(gold)
    row = next(r for r in pred[0].rows if len(r.packs) > 1 and r.gaps)
    row.packs.pop(0)
    row.packs[0].sku = "UNKNOWN"
    row.gaps.pop(0)
    o = evaluate(pred, load(gold)).overall
    assert o.facing_abs_err == 1
    assert o.sku_correct == o.packs - 2
    assert o.gaps_found == o.gaps - 1


def test_missing_prediction_scores_as_empty(gold):
    g = load(gold)
    rep = evaluate(g[1:], g)
    assert rep.missing == [frame_key(g[0].frame_ref)]
    assert rep.overall.sku_correct == rep.overall.packs - sum(len(r.packs) for r in g[0].rows)


def test_frame_key_pairs_run_paths_with_truth_paths():
    assert frame_key("runs/synth01/images/0003_G1-L-04.jpg") == "0003_G1-L-04.jpg"
    assert frame_key("images/0003_G1-L-04.jpg") == "0003_G1-L-04.jpg"
    assert frame_key("C:\\x\\walk.mp4#t=60.0") == "walk.mp4"


def test_script_runs_from_command_line(gold):
    args = [sys.executable, "tools/eval_readings.py", "--pred", str(gold), "--gold", str(gold)]
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    assert "SKU accuracy        100.0%" in res.stdout
