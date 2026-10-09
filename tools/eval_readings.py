"""Score predicted BayReadings against ground truth.

    python tools/eval_readings.py --pred runs/p01/bay_readings.jsonl --gold runs/synth01/truth.jsonl

Readings are paired by the file name in frame_ref, so a prediction made from
runs/synth01/images/0003_G1-L-04.jpg matches the truth line for images/0003_G1-L-04.jpg;
--by-bay pairs by bay_id instead (one reading per bay, e.g. a robot pass vs its truth.jsonl).
A gold reading with no prediction scores as an empty prediction.

- facing-count error: mean |pred packs - gold packs| over every gold row.
- SKU accuracy: gold packs whose x-matched prediction (same row, x-IoU >= 0.5) has the same sku.
- gap recall: gold gaps covered by a predicted gap with x-IoU >= 0.5.
- depth_left: over x-matched packs whose gold depth_left is known: share exactly right (a null
  prediction counts as wrong) and mean absolute error over the non-null ones. Shown only when
  the gold has depth.
- labels (robot readings): gold labels found in the same row with the same sku within 3 cm of x
  (recall), share of predicted labels that match one (precision), and matched prices exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from any folder

from shelfpulse.contracts import BayReading, parse_bay_reading  # noqa: E402

MATCH_IOU = 0.5


@dataclass
class Score:
    rows: int = 0
    facing_abs_err: int = 0
    packs: int = 0
    sku_correct: int = 0
    gaps: int = 0
    gaps_found: int = 0
    depth_n: int = 0  # matched packs with a gold depth_left
    depth_exact: int = 0
    depth_read: int = 0  # ... where the prediction has a depth_left
    depth_abs_err: int = 0
    labels: int = 0  # gold labels
    labels_pred: int = 0
    labels_found: int = 0
    labels_price_ok: int = 0

    def add(self, other: Score) -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    @property
    def facing_err(self) -> float:
        return self.facing_abs_err / self.rows if self.rows else 0.0

    @property
    def sku_acc(self) -> float | None:
        return self.sku_correct / self.packs if self.packs else None

    @property
    def gap_recall(self) -> float | None:
        return self.gaps_found / self.gaps if self.gaps else None

    @property
    def depth_acc(self) -> float | None:
        return self.depth_exact / self.depth_n if self.depth_n else None

    @property
    def depth_mae(self) -> float | None:
        return self.depth_abs_err / self.depth_read if self.depth_read else None


@dataclass
class Report:
    per_bay: list[tuple[str, str, Score]] = field(default_factory=list)  # (frame, bay_id, score)
    overall: Score = field(default_factory=Score)
    missing: list[str] = field(default_factory=list)  # gold frames with no prediction


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match(gold: list[tuple[float, float]], pred: list[tuple[float, float]]) -> list[int | None]:
    """Greedy one-to-one matching by x-IoU. Returns, per gold interval, the pred index or None."""
    pairs = sorted(
        ((iou(*g, *p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
        reverse=True,
    )
    out: list[int | None] = [None] * len(gold)
    used = set()
    for v, gi, pi in pairs:
        if v < MATCH_IOU:
            break
        if out[gi] is None and pi not in used:
            out[gi] = pi
            used.add(pi)
    return out


def score_bay(pred: BayReading | None, gold: BayReading) -> Score:
    s = Score()
    pred_rows = {r.row: r for r in pred.rows} if pred else {}
    for g in gold.rows:
        p = pred_rows.get(g.row)
        p_packs = p.packs if p else []
        p_gaps = p.gaps if p else []
        s.rows += 1
        s.facing_abs_err += abs(len(p_packs) - len(g.packs))

        m = match(
            [(k.x_cm, k.x_cm + k.w_cm) for k in g.packs],
            [(k.x_cm, k.x_cm + k.w_cm) for k in p_packs],
        )
        s.packs += len(g.packs)
        s.sku_correct += sum(
            1
            for k, pi in zip(g.packs, m, strict=True)
            if pi is not None and p_packs[pi].sku == k.sku
        )
        for k, pi in zip(g.packs, m, strict=True):
            if pi is None or k.depth_left is None:
                continue
            s.depth_n += 1
            got = p_packs[pi].depth_left
            if got is not None:
                s.depth_read += 1
                s.depth_abs_err += abs(got - k.depth_left)
                s.depth_exact += got == k.depth_left

        p_labels = list(p.labels) if p else []
        s.labels += len(g.labels)
        s.labels_pred += len(p_labels)
        for gl in g.labels:
            hit = next(
                (pl for pl in p_labels if pl.sku == gl.sku and abs(pl.x_cm - gl.x_cm) <= 3), None
            )
            if hit is not None:
                p_labels.remove(hit)
                s.labels_found += 1
                s.labels_price_ok += abs(hit.price - gl.price) < 0.005

        m = match(
            [(k.x_cm, k.x_cm + k.w_cm) for k in g.gaps], [(k.x_cm, k.x_cm + k.w_cm) for k in p_gaps]
        )
        s.gaps += len(g.gaps)
        s.gaps_found += sum(pi is not None for pi in m)
    return s


def frame_key(frame_ref: str) -> str:
    return frame_ref.replace("\\", "/").rsplit("/", 1)[-1].split("#", 1)[0]


def load(path: Path) -> list[BayReading]:
    with open(path, encoding="utf-8") as f:
        return [parse_bay_reading(json.loads(s)) for s in f if s.strip()]


def evaluate(pred: list[BayReading], gold: list[BayReading], by_bay: bool = False) -> Report:
    key_of = (lambda r: r.bay_id) if by_bay else (lambda r: frame_key(r.frame_ref))
    by_frame = {key_of(r): r for r in pred}
    rep = Report()
    for g in gold:
        key = key_of(g)
        p = by_frame.get(key)
        if p is None:
            rep.missing.append(key)
        s = score_bay(p, g)
        rep.per_bay.append((key, g.bay_id, s))
        rep.overall.add(s)
    return rep


def _pct(v: float | None, num: int, den: int) -> str:
    return "      n/a" if v is None else f"{v:6.1%} ({num}/{den})"


def format_report(rep: Report, per_bay: bool = True) -> str:
    lines = []
    if per_bay:
        lines.append(f"{'frame':<28} {'bay':<8} {'facing_err':>10}  {'sku_acc':<18} gap_recall")
        for frame, bay, s in rep.per_bay:
            lines.append(
                f"{frame:<28} {bay:<8} {s.facing_err:>10.2f}  "
                f"{_pct(s.sku_acc, s.sku_correct, s.packs):<18} "
                f"{_pct(s.gap_recall, s.gaps_found, s.gaps)}"
            )
        lines.append("")
    o = rep.overall
    lines.append(f"OVERALL  {len(rep.per_bay)} bays, {len(rep.missing)} with no prediction")
    lines.append(f"  facing-count error  {o.facing_err:.2f} packs per row")
    lines.append(f"  SKU accuracy        {_pct(o.sku_acc, o.sku_correct, o.packs).strip()}")
    lines.append(f"  gap recall          {_pct(o.gap_recall, o.gaps_found, o.gaps).strip()}")
    if o.labels:
        prec = o.labels_found / o.labels_pred if o.labels_pred else 0.0
        lines.append(
            "  labels found        "
            f"{_pct(o.labels_found / o.labels, o.labels_found, o.labels).strip()}"
            f"  (precision {prec:.1%}, prices exact {o.labels_price_ok}/{o.labels_found})"
        )
    if o.depth_n:
        mae = "n/a" if o.depth_mae is None else f"{o.depth_mae:.2f}"
        lines.append(
            f"  depth_left exact    {_pct(o.depth_acc, o.depth_exact, o.depth_n).strip()}"
            f"  (mean abs error {mae} packs over {o.depth_read} read)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--summary", action="store_true", help="overall numbers only")
    ap.add_argument("--by-bay", action="store_true", help="pair readings by bay_id")
    args = ap.parse_args(argv)
    rep = evaluate(load(args.pred), load(args.gold), args.by_bay)
    print(format_report(rep, per_bay=not args.summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
