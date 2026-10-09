"""SKU identification accuracy on synthetic shelves built from the real gallery.

    python -m tools.eval_identify [--n 40] [--seed 3] [--no-hints]

Packs are cut out with the truth boxes (an oracle detector), so this measures identification
alone. Two settings:
- seen:    shelves drawn from the same photos the index holds (an optimistic upper bound);
- heldout: for SKUs with 2+ photos one photo is hidden from the index and only that photo is
           drawn, so every identified pack is a photo the index never saw. SKUs without a
           held-out photo are drawn as plain coloured boxes, which should come out UNKNOWN.
Each runs with and without --misplace (a pack in the wrong slot tests the location hint).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

from shelfpulse.config import load_yaml
from shelfpulse.contracts import UNKNOWN_SKU, is_ambiguous, load_sku_master
from shelfpulse.perception.analyze import load_identify_settings
from shelfpulse.perception.embedder import Embedder
from shelfpulse.perception.identify import Identifier, PlanHints
from shelfpulse.perception.sku_index import SkuIndex, gallery_photos
from tools.synth.make import PX_PER_CM, load_geometry, make


def split_gallery(photos: dict[str, list[Path]]) -> tuple[dict, dict]:
    """(index part, held-out part): the last photo of every SKU with 2+ photos is held out."""
    index, held = {}, {}
    for sku, ps in photos.items():
        if len(ps) >= 2:
            index[sku], held[sku] = ps[:-1], ps[-1:]
        else:
            index[sku] = ps
    return index, held


def copy_gallery(photos: dict[str, list[Path]], root: Path) -> Path:
    for sku, ps in photos.items():
        (root / sku).mkdir(parents=True, exist_ok=True)
        for p in ps:
            shutil.copy(p, root / sku / p.name)
    return root


def crops_and_candidates(ident: Identifier, out: Path, truths) -> list:
    """[(truth reading, row, truth pack, gallery candidates)] using the truth boxes."""
    geo = load_geometry()
    items = []
    for t in truths:
        img = cv2.imread(str(out / t.frame_ref))
        packs, crops = [], []
        for r in t.rows:
            y1 = geo.y_px(geo.floor_cm(r.row))
            for p in r.packs:
                y0 = y1 - round(p.h_cm * PX_PER_CM)
                x0, x1 = round(p.x_cm * PX_PER_CM), round((p.x_cm + p.w_cm) * PX_PER_CM)
                packs.append((r.row, p))
                crops.append(img[y0:y1, x0:x1])
        cands = ident.candidates(crops)
        items += [(t, row, p, c) for (row, p), c in zip(packs, cands, strict=True)]
    return items


def score(ident: Identifier, items: list, drawn: set[str], plan: PlanHints, use_hints: bool):
    """Counts per kind: photo / box packs, plus misplaced photo packs (not the planned SKU)."""
    c: Counter = Counter()
    hints = ident.hints
    ident.hints = hints if use_hints else None
    for t, row, p, cands in items:
        sku, _ = ident.decide(cands, t.bay_id, row, p.x_cm, p.w_cm)
        planned = plan.planned(t.bay_id, row, p.x_cm + p.w_cm / 2)
        kinds = ["photo" if p.sku in drawn else "box"]
        if kinds[0] == "photo" and planned and p.sku != planned:
            kinds.append("misplaced")
            c["misplaced", "as_planned"] += sku == planned  # hint hid the misplacement
        for kind in kinds:
            c[kind, "n"] += 1
            if sku == p.sku:
                c[kind, "correct"] += 1
            elif sku == UNKNOWN_SKU:
                c[kind, "unknown"] += 1
            elif is_ambiguous(sku):
                c[kind, "ambiguous"] += 1
            else:
                c[kind, "wrong"] += 1
    ident.hints = hints
    return c


def report(name: str, c: Counter) -> str:
    def pct(k, kind):
        n = c[kind, "n"]
        return f"{c[kind, k] / n:6.1%}" if n else "   n/a"

    n = c["photo", "n"]
    line = (
        f"{name:<28} photo packs {n:5}: correct {pct('correct', 'photo')}  "
        f"ambiguous {pct('ambiguous', 'photo')}  unknown {pct('unknown', 'photo')}  "
        f"wrong {pct('wrong', 'photo')}"
    )
    if c["box", "n"]:
        line += f" | plain boxes {c['box', 'n']}: UNKNOWN {pct('unknown', 'box')}"
    if c["misplaced", "n"]:
        line += (
            f"\n{'':<28} misplaced packs {c['misplaced', 'n']}: "
            f"correct {pct('correct', 'misplaced')}  "
            f"read as the planned SKU {pct('as_planned', 'misplaced')}"
        )
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gallery", type=Path, default=Path("data/gallery"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-hints", action="store_true", help="also report without location hints")
    args = ap.parse_args(argv)

    cfg = load_yaml("perception")["identify"]
    skus = load_sku_master()
    emb = Embedder(cfg["backend"], cfg.get("model"), cfg["input_px"])
    photos = gallery_photos(args.gallery)
    if not photos:
        print(f"error: no gallery photos in {args.gallery}", file=sys.stderr)
        return 1
    index_part, held = split_gallery(photos)
    hints = PlanHints(cfg["planogram_dirs"])
    settings = load_identify_settings()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        held_root = copy_gallery(held, tmp / "held")
        setups = [
            ("seen", SkuIndex.build(photos, emb, skus), args.gallery, set(photos)),
            ("heldout", SkuIndex.build(index_part, emb, skus), held_root, set(held)),
        ]
        for name, index, gallery, drawn in setups:
            ident = Identifier(emb, index, skus, hints, settings)
            for misplace in (False, True):
                out = tmp / f"{name}_{misplace}"
                truths = make(
                    out, args.n, args.seed, misplaced=misplace, gallery=gallery, gaps=True
                )
                label = f"{name}{' +misplaced' if misplace else ''}"
                items = crops_and_candidates(ident, out, truths)
                print(report(label, score(ident, items, drawn, hints, True)))
                if args.no_hints:
                    print(report(f"{label} (no hints)", score(ident, items, drawn, hints, False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
