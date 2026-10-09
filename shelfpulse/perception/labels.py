"""Shelf-edge price labels on robot images -> contract labels [{x_cm, sku, price}] per row.

Only robot images carry labels (contract: always [] from cameras): they are high resolution and
taken close up. Each shelf rail is cut out at full resolution and read with OCR (RapidOCR, ONNX,
CPU). Text boxes on a rail are grouped into tags by x position; per tag:
- price: the first number like 89, 89.00 or 89,00 (an "Rs" / rupee prefix is fine);
- sku: an exact sku_id in the text, else the SKU whose pack_keywords match the most words
  (at least two, and strictly more than any other SKU: "Toor Dal 1 kg" -> DAL_1KG, not DAL_500G).
A tag with no recognisable SKU or no price is dropped (never guessed). x_cm is the tag's left edge.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import numpy as np

from shelfpulse.contracts import BAY_WIDTH_CM, Label, SkuRow
from shelfpulse.perception.shelves import Shelf

PRICE_RE = re.compile(r"(\d{1,6}(?:[.,]\d{1,2})?)")
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class LabelSettings:
    tag_gap_cm: float = 3.0  # text boxes further apart than this belong to different tags
    min_text_conf: float = 0.5
    min_keyword_hits: int = 2  # pack_keywords that must appear to name a SKU from words


@dataclass(frozen=True)
class TextBox:
    x0: float  # px in the rail crop
    x1: float
    y0: float
    text: str
    conf: float


DEFAULT_LABELS = LabelSettings()


class LabelReader:
    def __init__(self, skus: dict[str, SkuRow], settings: LabelSettings = DEFAULT_LABELS):
        self.skus = skus
        self.s = settings
        self._ocr = None

    def _engine(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR  # Vision-only dependency, loaded on use

            self._ocr = RapidOCR()
        return self._ocr

    def ocr(self, crop: np.ndarray) -> list[TextBox]:
        result, _ = self._engine()(crop)
        out = []
        for box, text, conf in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            if float(conf) >= self.s.min_text_conf:
                out.append(TextBox(min(xs), max(xs), min(ys), str(text), float(conf)))
        return out

    def sku_of(self, text: str) -> str | None:
        upper = text.upper().replace(" ", "")
        exact = [s for s in self.skus if s in upper]
        if exact:
            return max(exact, key=len)  # RICE_5KG over a shorter accidental match
        words = set(WORD_RE.findall(text.lower()))
        compact = "".join(WORD_RE.findall(text.lower()))  # "150 g" -> "150g"
        scores = []
        for sku_id, row in self.skus.items():
            kws = [k.lower() for k in row.pack_keywords]
            scores.append((sum(k in words or k in compact for k in kws), sku_id))
        scores.sort(reverse=True)
        (hits, best), runner_up = scores[0], (scores[1][0] if len(scores) > 1 else 0)
        # Must hit enough keywords and beat every other SKU outright (never guess a tie).
        return best if hits >= self.s.min_keyword_hits and hits > runner_up else None

    @staticmethod
    def price_of(text: str) -> float | None:
        m = PRICE_RE.search(text.replace(" ", ""))
        return float(m.group(1).replace(",", ".")) if m else None

    def read_rail(self, crop: np.ndarray, px_per_cm: float) -> list[Label]:
        boxes = sorted(self.ocr(crop), key=lambda b: b.x0)
        tags: list[list[TextBox]] = []
        for b in boxes:
            if tags and b.x0 - max(t.x1 for t in tags[-1]) < self.s.tag_gap_cm * px_per_cm:
                tags[-1].append(b)
            else:
                tags.append([b])
        labels = []
        for tag in tags:
            lines = sorted(tag, key=lambda b: b.y0)
            sku = next((s for b in lines if (s := self.sku_of(b.text))), None)
            sku = sku or self.sku_of(" ".join(b.text for b in lines))
            # Price from a line other than the name, so digits in "RICE_5KG" don't count.
            prices = [
                p for b in lines if self.sku_of(b.text) is None and (p := self.price_of(b.text))
            ]
            if sku is None or not prices:
                continue
            x_cm = min(max(min(b.x0 for b in tag) / px_per_cm, 0.0), BAY_WIDTH_CM)
            labels.append(Label(x_cm=round(x_cm, 1), sku=sku, price=prices[0]))
        return labels

    def read(
        self, image: np.ndarray, shelves: list[Shelf], shelf_px_per_cm: float, px_per_cm: float
    ) -> dict[int, list[Label]]:
        """Labels per row. `shelves` were found at shelf_px_per_cm; `image` is at px_per_cm."""
        k = px_per_cm / shelf_px_per_cm
        out = {}
        for s in shelves:
            y0, y1 = round(s.rail.y0 * k), round(s.rail.y1 * k)
            pad = round(0.5 * px_per_cm)
            crop = image[max(y0 - pad, 0) : y1 + pad]
            try:
                out[s.row] = self.read_rail(crop, px_per_cm)
            except Exception as e:  # OCR must never break a robot run
                print(f"warning: label OCR failed on row {s.row}: {e}", file=sys.stderr)
                out[s.row] = []
        return out
