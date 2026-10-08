"""Train a YOLO pack detector on SKU-110K. Run on a free Colab/Kaggle GPU, not on a laptop.

Colab (Runtime -> Change runtime type -> T4 GPU), one cell each:
    !git clone https://github.com/sanyamagarwal8055/Shelfish.git
    %cd Shelfish
    !git checkout vision/dev
    !pip install -q -r requirements.txt ultralytics
    !python training/train_sku110k.py --epochs 50
    !python training/eval_sku110k.py --weights runs/train/sku110k/weights/best.pt

The first run downloads SKU-110K (several GB) into data/raw/ (git-ignored). Copy best.pt to Drive,
share it with Sanyam, and on your laptop put it at data/models/sku110k_yolo.pt (or point
detector.weights in configs/perception.yaml at it). Never commit weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def use_data_raw() -> None:
    """Make Ultralytics download and look for datasets under data/raw/."""
    from ultralytics import settings

    raw = REPO / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    settings.update({"datasets_dir": str(raw)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt", help="start from these COCO weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu' for a smoke test")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on part of the data")
    args = ap.parse_args(argv)

    use_data_raw()
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data="SKU-110K.yaml",
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
