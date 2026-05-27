from __future__ import annotations

import torch
from PIL import Image

from ._hf_model import AutoProcessorModelBase


class Gemma(AutoProcessorModelBase):
    display_name = "Gemma"
    default_model_id = "google/paligemma-3b-mix-224"

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.model_id = model_id or self.default_model_id
        self.name = self._name_from_model_id(self.model_id)
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
        self._model_cls = model_cls

        self._load_input_artifact_and_model()

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        return str(model_id).rsplit("/", 1)[-1].lower().replace("_", "-")

    def _load_model(self):
        return self._load_with_cache_first(
            self._model_cls.from_pretrained,
            self.model_id,
            "model",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def predict(self, image: Image.Image, prompt: str) -> str:
        prompt = self._prepare_prompt(prompt)
        input_device = next(self.model.parameters()).device
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(input_device)

        gen_kwargs = self._sampling_generation_kwargs()

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        if "input_ids" in inputs:
            generated = output[0][inputs["input_ids"].shape[-1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()

        return self.processor.decode(output[0], skip_special_tokens=True).strip()
