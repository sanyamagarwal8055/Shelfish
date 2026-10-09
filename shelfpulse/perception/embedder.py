"""Pack crop -> L2-normalised embedding ("fingerprint"), behind one small class.

Backends: "dinov2" (default, facebook/dinov2-small, 384-d CLS token) or "clip"
(openai/clip-vit-base-patch32 image features). Crops are resized straight to input_px x input_px
(no centre crop: tall packs keep their top and bottom) with the model's own normalisation, so
gallery photos and shelf crops go through exactly the same steps.
"""

from __future__ import annotations

import cv2
import numpy as np

BACKENDS = {"dinov2": "facebook/dinov2-small", "clip": "openai/clip-vit-base-patch32"}


class Embedder:
    def __init__(
        self,
        backend: str = "dinov2",
        model: str | None = None,
        input_px: int = 224,
        batch: int = 32,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"embedder backend must be one of {list(BACKENDS)}, got {backend!r}")
        import torch  # heavy imports only when an embedder is built
        from transformers import AutoImageProcessor, AutoModel, CLIPModel

        self.backend = backend
        self.model_name = model or BACKENDS[backend]
        self.input_px = input_px
        self.batch = batch
        self._torch = torch
        proc = AutoImageProcessor.from_pretrained(self.model_name)
        self._mean = np.asarray(proc.image_mean, np.float32)
        self._std = np.asarray(proc.image_std, np.float32)
        if backend == "clip":
            self._model = CLIPModel.from_pretrained(self.model_name).eval()
        else:
            self._model = AutoModel.from_pretrained(self.model_name).eval()

    @property
    def name(self) -> str:
        return f"{self.model_name}@{self.input_px}"

    def _prep(self, crops: list[np.ndarray]):
        px = self.input_px
        rgb = [cv2.cvtColor(cv2.resize(c, (px, px), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2RGB) for c in crops]  # fmt: skip
        x = (np.stack(rgb).astype(np.float32) / 255.0 - self._mean) / self._std
        return self._torch.from_numpy(x.transpose(0, 3, 1, 2).copy())

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops (any size) -> (n, d) float32, each row of unit length."""
        if not crops:
            return np.zeros((0, 0), np.float32)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(crops), self.batch):
                x = self._prep(crops[i : i + self.batch])
                if self.backend == "clip":
                    v = self._model.get_image_features(pixel_values=x)
                    v = v if isinstance(v, self._torch.Tensor) else v.pooler_output
                else:
                    v = self._model(pixel_values=x).pooler_output
                out.append(v.cpu().numpy().astype(np.float32))
        v = np.concatenate(out)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
