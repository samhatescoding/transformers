from __future__ import annotations

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from ._hf_model import HuggingFaceModelBase


class LlavaNextModelBase(HuggingFaceModelBase):
    display_name = "LLaVA-NeXT"
    model_loader = LlavaNextForConditionalGeneration.from_pretrained
    model_torch_dtype = torch.float16
    default_model_id: str

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.name = self.model_id.rsplit("/", 1)[-1].lower().replace("_", "-")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._load_input_artifact_and_model()

    def _load_input_artifact(self):
        return self._load_with_cache_first(
            LlavaNextProcessor.from_pretrained,
            self.model_id,
            "processor",
        )

    def predict(self, image: Image.Image, prompt: str) -> str:
        instruction = self._extract_prompt(prompt)
        rendered = f"USER: <image>\n{instruction}\nASSISTANT:"
        inputs = self.processor(
            text=rendered,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model.generate(**inputs, **self._sampling_generation_kwargs())
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
