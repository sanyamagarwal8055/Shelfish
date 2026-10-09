"""Synthetic YOLO training data: valid labels, held-out photos only in validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.synth.yolo_export import export, split_photos


def test_split_holds_out_one_photo_per_multi_photo_sku():
    photos = {"A": [Path("a1"), Path("a2"), Path("a3")], "B": [Path("b1")]}
    train, val = split_photos(photos)
    assert train["A"] == [Path("a1"), Path("a2")] and val["A"] == [Path("a3")]
    assert train["B"] == val["B"] == [Path("b1")]  # a single photo can't be held out


def test_export_writes_valid_yolo_labels(tmp_path):
    gallery = Path(__file__).resolve().parents[2] / "data" / "gallery"
    stats = export(tmp_path, n_train=3, n_val=2, seed=4, gallery=gallery)
    assert stats["train_boxes"] > 20 and stats["val_boxes"] > 10
    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "object"} and Path(data["path"]).is_dir()
    for split, n in (("train", 3), ("val", 2)):
        images = sorted((tmp_path / "images" / split).glob("*.jpg"))
        labels = sorted((tmp_path / "labels" / split).glob("*.txt"))
        assert len(images) == len(labels) == n
        for lb in labels:
            for line in lb.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                cls, *xywh = line.split()
                assert cls == "0" and len(xywh) == 4
                cx, cy, w, h = map(float, xywh)
                assert 0 < w <= 1 and 0 < h <= 1
                assert 0 <= cx - w / 2 + 1e-6 and cx + w / 2 <= 1 + 1e-6
                assert 0 <= cy - h / 2 + 1e-6 and cy + h / 2 <= 1 + 1e-6
