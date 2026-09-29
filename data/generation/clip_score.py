from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image


@dataclass
class ClipScorer:
    """Lazy CLIP scorer; the heavy model is loaded only when the stage is used."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None

    def __post_init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device = self.device or os.environ.get("CLIP_DEVICE", "cpu")

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("CLIP scoring requires torch and transformers") from exc
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device).eval()

    @staticmethod
    def _features(value: Any) -> Any:
        """Normalize CLIP API differences across Transformers 4.x and 5.x."""
        if hasattr(value, "pooler_output"):
            return value.pooler_output
        if isinstance(value, tuple):
            return value[0]
        return value

    def score(self, image_url: str, text: str, *, timeout: float = 30) -> float:
        if image_url.startswith("file://"):
            from pathlib import Path
            image = Image.open(Path(image_url.removeprefix("file://"))).convert("RGB")
        else:
            response = requests.get(image_url, timeout=timeout, headers={"User-Agent": "SightlineDataBuilder/0.1"})
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        self._load()
        import torch
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            image_features = self._features(self._model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = self._features(self._model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            ))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Cosine similarity is used directly. A one-item softmax would
            # always return 1.0 and is not a useful relevance score.
            value = (image_features * text_features).sum(dim=-1)[0].item()
        return float(value)
