from __future__ import annotations

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from .base import BaseModel


class Falcon(BaseModel):
    def __init__(
        self,
        model_id: str = "tiiuae/falcon-11B-vlm",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.name = "falcon"
        self.model_id = "tiiuae/falcon-11B-vlm" if model_id == "falcon" else model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if not torch.cuda.is_available() and self.model_id == "tiiuae/falcon-11B-vlm":
            raise RuntimeError(
                "falcon-11B-vlm on CPU is likely to terminate due to RAM limits during weight "
                "materialization. Run on CUDA GPU or pass a smaller model_id."
            )

        print("Loading Falcon VLM processor...")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)

        print("Loading Falcon VLM model...")
        if torch.cuda.is_available():
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            # Avoid meta-device initialization on CPU-only setups.
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            self.model.to("cpu")

    @staticmethod
    def _adapt_prompt(prompt: str) -> str:
        p = prompt.strip()
        if "<image>" not in p:
            p = f"USER: <image>\n{p}\nASSISTANT:"
        return p

    def predict(self, image: Image.Image, prompt: str) -> str:
        prompt = self._adapt_prompt(prompt)
        # Pick a concrete device where weights are materialized.
        input_device = next(self.model.parameters()).device
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(input_device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
