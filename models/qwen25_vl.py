from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ._base_model import BaseModel


class Qwen25VL(BaseModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct" if model_id == "qwen25-vl" else model_id
        self.name = self._name_from_model_id(self.model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print("Loading Qwen2.5-VL processor...")
        self.processor = self._load_processor()

        print("Loading Qwen2.5-VL model...")
        self.model = self._load_model()

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        normalized = str(model_id or "").strip()
        if normalized in {"qwen25-vl", "Qwen/Qwen2.5-VL-3B-Instruct"}:
            return "qwen25-vl"
        if normalized == "Qwen/Qwen2.5-VL-72B-Instruct":
            return "qwen25-vl-72b"
        return normalized.rsplit("/", 1)[-1].lower().replace(".", "").replace("_", "-")

    def _load_processor(self):
        try:
            return AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:
            print(f"Falling back to cached Qwen2.5-VL processor after load error: {exc}")
            return AutoProcessor.from_pretrained(self.model_id, local_files_only=True)

    def _load_model(self):
        if torch.cuda.is_available():
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
        else:
            load_kwargs = {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": False,
            }

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        except Exception as exc:
            print(f"Falling back to cached Qwen2.5-VL model after load error: {exc}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                local_files_only=True,
                **load_kwargs,
            )

        if not torch.cuda.is_available():
            model.to("cpu")
        return model

    @staticmethod
    def _adapt_prompt(prompt: str) -> str:
        return (
            str(prompt or "")
            .replace("USER:", "")
            .replace("ASSISTANT:", "")
            .replace("<image>", "")
            .strip()
            or "Describe the image."
        )

    def predict(self, image: Image.Image, prompt: str) -> str:
        prompt = self._adapt_prompt(prompt)
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
