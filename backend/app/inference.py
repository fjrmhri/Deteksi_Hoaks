"""Model loading and inference utilities for the FastAPI service."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_settings
from .text_utils import normalise_text


class HoaxDetector:
    """Wrapper around an IndoBERT sequence classification model."""

    def __init__(self, model_name_or_path: str, device: str = "auto", max_length: int = 256) -> None:
        target_device = device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(target_device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        # Prefer label mapping stored with the fine-tuned model
        if self.model.config.id2label:
            self.id2label: Dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
        else:
            self.id2label = {0: "not_hoax", 1: "hoax"}

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        """Return the predicted label and its probability for ``text``."""

        processed = normalise_text(text)
        encoded = self.tokenizer(
            processed,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        outputs = self.model(**encoded)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        score, label_id = torch.max(probs, dim=-1)
        label_name = self.id2label.get(label_id.item(), str(label_id.item()))
        return label_name, float(score.item())


@lru_cache(maxsize=1)
def get_detector() -> HoaxDetector:
    """Load (and memoise) the detector instance."""

    settings = get_settings()
    return HoaxDetector(
        model_name_or_path=settings.model_name_or_path,
        device=settings.device,
        max_length=settings.max_length,
    )
