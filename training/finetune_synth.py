"""Fine-tune the SKU-110K detector on synthetic small-pack shelves + part of SKU-110K, then compare.

Kaggle notebook (GPU T4 x2, Internet on). Add the current weights as a dataset input (upload
best.pt or best.zip - it is the same file), one cell, then Save Version -> Save & Run All:
    !git clone https://github.com/sanyamagarwal8055/Shelfish.git /tmp/Shelfish
    %cd /tmp/Shelfish
    !pip install -q -r requirements.txt ultralytics faiss-cpu
    !python training/finetune_synth.py --weights /kaggle/input/<your-dataset>/best.zip
    !mkdir -p /kaggle/working/out && cp runs/train/sku110k_synth/weights/best.pt \\
        runs/eval/finetune_compare.json /kaggle/working/out/

Steps: (1) tools.synth.yolo_export writes synthetic shelves from data/gallery (validation uses
held-out photos only); (2) SKU-110K is downloaded if needed and a random subset of its train
images is mixed in so the model keeps its real-shelf skill; (3) fine-tune from --weights;
(4) the old and new weights are both measured on the SKU-110K test split and the synthetic
held-out set, saved to runs/eval/finetune_compare.json. Keep the new weights only if SKU-110K
mAP@0.5 holds up (>= the old) and the synthetic score improves.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_sku110k import REPO, sku110k_data  # noqa: E402


def measure(weights: str, data: str, split: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    m = YOLO(weights).val(
        data=data,
        split=split,
        imgsz=imgsz,
        batch=16,
        max_det=1000,
        project=str(REPO / "runs" / "eval"),
        name="cmp",
        exist_ok=True,
        verbose=False,
    )
    return {
        "map50": round(float(m.box.map50), 4),
        "map50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--weights", required=True, type=Path, help="current SKU-110K best.pt (or .zip)"
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--synth-train", type=int, default=1500)
    ap.add_argument("--synth-val", type=int, default=300)
    ap.add_argument("--sku-train", type=int, default=3000, help="SKU-110K train images mixed in")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--optimizer", default="AdamW", help="explicit, so --lr0 is honoured")
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (0 = none)")
    ap.add_argument("--name", default="sku110k_synth", help="run name under runs/train/")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    from tools.synth.yolo_export import export

    old = REPO / "runs" / "old_best.pt"  # Kaggle serves the upload as .zip; YOLO wants .pt
    old.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.weights, old)

    synth = REPO / "data" / "raw" / "synth_yolo"
    if not (synth / "data.yaml").is_file():
        print("synthetic set:", export(synth, args.synth_train, args.synth_val, args.seed))
    sku_yaml = sku110k_data()
    sku = check_det_dataset(sku_yaml)  # downloads SKU-110K on first use
    lines = Path(sku["train"]).read_text(encoding="utf-8").splitlines()
    root = Path(sku["train"]).parent
    random.Random(args.seed).shuffle(lines)
    subset = REPO / "runs" / "sku110k_train_subset.txt"
    picked = [str((root / ln).resolve()) if ln.startswith("./") else ln for ln in lines]
    subset.write_text("\n".join(picked[: args.sku_train]) + "\n", encoding="utf-8")
    mix = REPO / "runs" / "mix_data.yaml"
    data = {
        "train": [str(synth / "images" / "train"), str(subset)],
        "val": [str(synth / "images" / "val"), str(sku["val"])],
        "names": {0: "object"},
    }
    mix.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    YOLO(str(old)).train(
        data=str(mix),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        freeze=args.freeze or None,
        project=str(REPO / "runs" / "train"),
        name=args.name,
        exist_ok=True,
    )
    new = REPO / "runs" / "train" / args.name / "weights" / "best.pt"

    result = {}
    for label, w in (("old", old), ("new", new)):
        result[label] = {
            "sku110k_test": measure(str(w), sku_yaml, "test", args.imgsz),
            "synthetic_heldout_val": measure(str(w), str(synth / "data.yaml"), "val", args.imgsz),
        }
    out = REPO / "runs" / "eval" / "finetune_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"new weights: {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
