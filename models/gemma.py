from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor

from ._base_model import BaseModel


class Gemma(BaseModel):
    def __init__(
        self,
        model_id: str = "google/paligemma-3b-mix-224",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.name = "gemma"
        self.model_id = "google/paligemma-3b-mix-224" if model_id == "gemma" else model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Import model classes lazily for compatibility across transformers versions.
        import transformers  # type: ignore

        model_cls = None
        for cls_name in (
            "PaliGemmaForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            model_cls = getattr(transformers, cls_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise ImportError(
                "No compatible vision-language model class found in transformers. "
                "Tried: PaliGemmaForConditionalGeneration, "
                "AutoModelForImageTextToText, AutoModelForVision2Seq."
            )

        print("Loading Gemma processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        print("Loading Gemma model...")
        if torch.cuda.is_available():
            self.model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            # Avoid cpu+disk offload path that can be extremely slow/unstable on some setups.
            self.model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            self.model.to("cpu")

    @staticmethod
    def _adapt_prompt(prompt: str) -> str:
        p = prompt.replace("<image>", "").strip()
        p = p.replace("USER:", "").replace("ASSISTANT:", "").strip()
        return p or "Describe the image."

    def predict(self, image: Image.Image, prompt: str) -> str:
        prompt = self._adapt_prompt(prompt)
        input_device = next(self.model.parameters()).device
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(input_device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        if "input_ids" in inputs:
            generated = output[0][inputs["input_ids"].shape[-1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()

        return self.processor.decode(output[0], skip_special_tokens=True).strip()
