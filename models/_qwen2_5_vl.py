from __future__ import annotations

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration

from ._hf_model import AutoProcessorModelBase


class _Qwen25VLBase(AutoProcessorModelBase):
    display_name = "Qwen2.5-VL"
    model_loader = Qwen2_5_VLForConditionalGeneration.from_pretrained
    model_id: str
    name: str

    def __init__(
        self,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._load_input_artifact_and_model()

    def predict(self, image: Image.Image, prompt: str) -> str:
        prompt = self._prepare_prompt(prompt)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(input_device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
