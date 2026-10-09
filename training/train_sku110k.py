"""Train a YOLO pack detector on SKU-110K. Run on a free Kaggle GPU, not on a laptop.

Kaggle notebook (Settings: GPU T4 x2, Internet on), one cell, then Save Version ->
Save & Run All (Commit); it keeps running with the tab closed:
    !git clone https://github.com/sanyamagarwal8055/Shelfish.git /tmp/Shelfish
    %cd /tmp/Shelfish
    !git checkout vision/dev
    !pip install -q -r requirements.txt ultralytics
    !python training/train_sku110k.py --epochs 30
    !python training/eval_sku110k.py --weights runs/train/sku110k/weights/best.pt
    !mkdir -p /kaggle/working/out
    !cp runs/train/sku110k/weights/best.pt runs/train/sku110k/results.csv \\
        runs/eval/sku110k_test.json /kaggle/working/out/

Work in /tmp: the dataset is ~25 GB unpacked and /kaggle/working holds 20 GB. The first run
downloads SKU-110K (~11 GB) into data/raw/SKU-110K/ (git-ignored). Share best.pt with Sanyam
over Drive; on your laptop put it at data/models/sku110k_yolo.pt. Never commit weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sku110k_data() -> str:
    """Ultralytics' SKU-110K.yaml with an absolute dataset root, written to runs/sku110k_data.yaml.

    Ultralytics reads its datasets_dir setting once at import, so changing the setting at runtime
    only takes effect in the *next* process (eval would then re-download 11 GB elsewhere). A fixed
    absolute path keeps download, training and eval on the same folder: data/raw/SKU-110K, or an
    existing <repo>/datasets/SKU-110K left by an earlier run.
    """
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    root = REPO / "data" / "raw" / "SKU-110K"
    legacy = REPO / "datasets" / "SKU-110K"
    if not root.is_dir() and legacy.is_dir():
        root = legacy
    root.parent.mkdir(parents=True, exist_ok=True)
    cfg = YAML.load(check_yaml("SKU-110K.yaml"))
    cfg["path"] = str(root)
    out = REPO / "runs" / "sku110k_data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(out, cfg)
    return str(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt", help="start from these COCO weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu' for a smoke test")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on part of the data")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=sku110k_data(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        project=str(REPO / "runs" / "train"),
        name="sku110k",
        exist_ok=True,
    )
    print(f"best weights: {REPO / 'runs' / 'train' / 'sku110k' / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
