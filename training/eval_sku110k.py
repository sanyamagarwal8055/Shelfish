"""Measure a pack detector's mAP@0.5 on the SKU-110K test split.

    python training/eval_sku110k.py --weights data/models/sku110k_yolo.pt

Prints the measured numbers and saves them to runs/eval/sku110k_test.json. Report these exact
numbers in the PR; never estimate them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script from any folder

from train_sku110k import REPO, use_data_raw  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="GPU index or 'cpu' (default: auto)")
    args = ap.parse_args(argv)

    use_data_raw()
    from ultralytics import YOLO

    m = YOLO(str(args.weights)).val(
        data="SKU-110K.yaml",
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        max_det=1000,  # SKU-110K images hold up to ~720 packs; the default 300 caps recall
        project=str(REPO / "runs" / "eval"),
        name="sku110k_test",
        exist_ok=True,
    )
    result = {
        "weights": str(args.weights),
        "split": "test",
        "imgsz": args.imgsz,
        "map50": round(float(m.box.map50), 4),
        "map50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out = REPO / "runs" / "eval" / "sku110k_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
