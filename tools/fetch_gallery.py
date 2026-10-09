"""Build data/gallery/<sku_id>/ from Open Food Facts product photos.

    python -m tools.fetch_gallery --candidates  # all photos -> data/raw/gallery_candidates/
    python -m tools.fetch_gallery               # kept fronts -> data/gallery/ + SOURCES.csv

Each SKU in data/sku_master.csv maps to one real product (one brand and size, like a real store
SKU). Open Food Facts uploads mix fronts, backs and nutrition panels, so `keep` lists the photo
ids that show the pack front, picked by hand from the candidates; "front" is the product's
selected front image. Non-food packs come from the sister sites (Open Beauty / Products Facts).
Photos are 400 px (well under 200 KB) and CC BY-SA: SOURCES.csv credits them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "Shelfish-hackathon/0.1 (student project; rampuriajainam@gmail.com)"
API = "https://world.{site}.org/api/v2/product/{code}.json?fields=product_name,brands,images"
IMG = "https://images.{site}.org/images/products/{path}/{name}.400.jpg"
SITE_NAMES = {
    "openfoodfacts": "Open Food Facts",
    "openbeautyfacts": "Open Beauty Facts",  # shampoo, soap, toothpaste
    "openproductsfacts": "Open Products Facts",  # detergent
}


@dataclass(frozen=True)
class Pick:
    code: str  # barcode on Open Food Facts
    product: str  # what it is, for SOURCES.csv
    keep: tuple[str, ...] = ("front",)  # photo ids showing the pack front
    site: str = "openfoodfacts"

    @property
    def licence(self) -> str:
        return f"CC BY-SA 3.0, {SITE_NAMES[self.site]} contributors"


PICKS: dict[str, Pick] = {
    "RICE_1KG": Pick("8901537074231", "Daawat Pulav Basmati Rice 1 kg", ("1",)),
    "RICE_5KG": Pick("8901537007116", "Daawat Rozana Basmati Rice 5 kg", ("2",)),  # front = back
    "DAL_1KG": Pick("8904043926216", "Tata Sampann Unpolished Toor Dal 1 kg", ("2",)),
    "DAL_500G": Pick("18902901224726", "Good Life Toor Dal 500 g", ("1",)),
    "ATTA_5KG": Pick("8906000210314", "Pillsbury Chakki Fresh Atta 5 kg", ("1", "2")),
    "SUGAR_1KG": Pick("8906009011301", "Parry's White Label Sugar 1 kg", ("8",)),
    "SALT_1KG": Pick("8904043901015", "Tata Salt 1 kg", ("1", "2", "6")),
    "OIL_1L": Pick("8906007280242", "Fortune Sunlite Refined Sunflower Oil 1 L", ("1", "7")),
    "TEA_250G": Pick("8901030456381", "Brooke Bond Taaza Tea 250 g", ("1",)),
    "COFFEE_100G": Pick("8901030774980", "Bru Green Label 100 g", ("1",)),
    "CHIPS_52G": Pick("8901491101837", "Lay's Classic Salted 50 g", ("1", "6")),
    "BISCUIT_200G": Pick("0901063139213", "Britannia Bourbon 150 g", ("front", "1")),
    "NAMKEEN_200G": Pick("8904004400731", "Haldiram's Aloo Bhujia 200 g", ("1", "10")),
    "COLA_750ML": Pick("3948764012273", "Coca-Cola 750 ml", ("front", "2")),
    "JUICE_1L": Pick("4796008920049", "Real Mixed Fruit Juice 1 L", ("1", "3", "6")),
    "WATER_1L": Pick("3948764082764", "Schweppes Packaged Drinking Water 1 L", ("1",)),
    "CORNFLAKES_475G": Pick("8901499008190", "Kellogg's Corn Flakes 475 g", ("1", "6")),
    "OATS_1KG": Pick("8902710100150", "Bagrry's White Oats 1 kg", ("1", "3")),
    "DETERGENT_1KG": Pick(
        "4987176069696", "Ariel Perfectwash 1 kg", ("1",), site="openproductsfacts"
    ),
    "SHAMPOO_180ML": Pick(
        "8901030929656", "Indulekha Bringha Shampoo 340 ml", ("1",), site="openbeautyfacts"
    ),
    "SOAP_100G": Pick(
        "8904132945999", "Get Real Sandalwood Soap 100 g", ("front", "1"), site="openbeautyfacts"
    ),
    "TOOTHPASTE_150G": Pick(
        "8901314644312", "Colgate Herbal Toothpaste 200 g", ("1",), site="openbeautyfacts"
    ),
}


def image_path(code: str) -> str:
    """Open Food Facts folder for a barcode: 8901499008190 -> 890/149/900/8190."""
    if len(code) <= 8:
        return code
    return f"{code[:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def photo_ids(pick: Pick) -> dict[str, str]:
    """{photo id: file name}: numbered uploads plus the selected front."""
    product = json.loads(_get(API.format(site=pick.site, code=pick.code)))["product"]
    images = product.get("images", {})
    out = {k: k for k in images if k.isdigit()}
    front = next((k for k in images if k.startswith("front")), None)
    if front:
        out["front"] = f"{front}.{images[front]['rev']}"
    return out


def fetch(sku: str, pick: Pick, folder: Path, ids: tuple[str, ...] | None) -> list[tuple[str, str]]:
    """Download photos of one pick into folder; returns (file, url) per photo saved."""
    folder.mkdir(parents=True, exist_ok=True)
    available = photo_ids(pick)
    wanted = available if ids is None else {i: available[i] for i in ids if i in available}
    missing = [] if ids is None else [i for i in ids if i not in available]
    if missing:
        print(f"warning: {sku} has no photo {missing}", file=sys.stderr)
    saved = []
    for pid, name in sorted(wanted.items()):
        url = IMG.format(site=pick.site, path=image_path(pick.code), name=name)
        out = folder / f"{sku}_{pid}.jpg"
        out.write_bytes(_get(url))
        saved.append((out.name, url))
        time.sleep(0.3)  # be gentle with a volunteer-run server
    return saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--candidates", action="store_true", help="download every photo for review")
    ap.add_argument(
        "--out", type=Path, help="default data/gallery (or data/raw/gallery_candidates)"
    )
    ap.add_argument("--skus", help="comma-separated subset")
    args = ap.parse_args(argv)

    root = args.out or Path("data/raw/gallery_candidates" if args.candidates else "data/gallery")
    skus = args.skus.split(",") if args.skus else list(PICKS)
    rows = []
    for sku in skus:
        pick = PICKS[sku]
        folder = root if args.candidates else root / sku
        saved = fetch(sku, pick, folder, None if args.candidates else pick.keep)
        rows += [(sku, f, pick.code, pick.product, url, pick.licence) for f, url in saved]
        print(f"{sku:16} {len(saved)} photo(s)  {pick.product}")
    if not args.candidates:
        with open(root / "SOURCES.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sku_id", "file", "barcode", "product", "url", "licence"])
            w.writerows(rows)
    print(f"wrote {len(rows)} photos to {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
