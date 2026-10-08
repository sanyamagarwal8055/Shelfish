"""Embed every gallery pack-shot and save the FAISS index analyze() names packs with.

    python scripts/build_gallery_index.py [--gallery data/gallery] [--out runs/gallery_index]

Defaults come from the identify section of configs/perception.yaml. Rebuild after adding photos
or changing identify.backend / model / input_px.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from any folder

from shelfpulse.config import REPO_ROOT, load_yaml  # noqa: E402
from shelfpulse.contracts import load_sku_master  # noqa: E402
from shelfpulse.perception.embedder import Embedder  # noqa: E402
from shelfpulse.perception.sku_index import SkuIndex, gallery_photos  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cfg = load_yaml("perception")["identify"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gallery", type=Path, default=Path(cfg["gallery"]))
    ap.add_argument("--out", type=Path, default=Path(cfg["index_dir"]))
    args = ap.parse_args(argv)

    gallery = args.gallery if args.gallery.is_absolute() else REPO_ROOT / args.gallery
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    photos = gallery_photos(gallery)
    if not photos:
        print(f"error: no photos in {gallery}/<sku_id>/", file=sys.stderr)
        return 1
    skus = load_sku_master()
    unknown = sorted(set(photos) - set(skus))
    if unknown:
        print(f"warning: gallery folders not in sku_master: {unknown}", file=sys.stderr)
    missing = sorted(set(skus) - set(photos))

    t = time.time()
    index = SkuIndex.build(
        photos, Embedder(cfg["backend"], cfg.get("model"), cfg["input_px"]), skus
    )
    index.save(out)
    n = sum(len(p) for p in photos.values())
    print(
        f"indexed {n} photos of {len(photos)} SKUs ({len(index.labels)} vectors, {index.model}) "
        f"in {time.time() - t:.0f}s -> {out}"
    )
    if missing:
        print(f"no photos yet for: {missing} (they will read as UNKNOWN or a look-alike)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
