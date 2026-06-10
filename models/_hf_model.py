from __future__ import annotations

from typing import Any, Callable

import torch
from PIL import Image
from transformers import AutoProcessor

from ._base_model import BaseModel


class CacheFirstLoaderMixin:
    def _load_with_cache_first(
        self,
        loader: Callable[..., Any],
        model_id: str,
        artifact_name: str,
        display_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        name = display_name or getattr(self, "display_name", None) or getattr(self, "name", self.__class__.__name__)
        try:
            return loader(model_id, local_files_only=True, **kwargs)
        except Exception as cache_exc:
            print(f"[INFO] Local cache load failed for {name} {artifact_name}: {cache_exc}")
            print(f"[INFO] Retrying {name} {artifact_name} load from hub.")
            return loader(model_id, **kwargs)


class HuggingFaceModelBase(CacheFirstLoaderMixin, BaseModel):
    display_name: str | None = None
    input_artifact_attr = "processor"
    input_artifact_name = "processor"
    model_loader: Callable[..., Any] | None = None
    model_torch_dtype = torch.bfloat16
    default_prompt = "Describe the image."

    def _get_display_name(self) -> str:
        return self.display_name or getattr(self, "name", self.__class__.__name__)

    def _load_input_artifact_and_model(self) -> None:
        name = self._get_display_name()
        self._log_loading_artifact(self.input_artifact_name, name)
        setattr(self, self.input_artifact_attr, self._load_input_artifact())

        self._log_loading_artifact("model", name)
        self.model = self._load_model()

    def _load_input_artifact(self) -> Any:
        raise NotImplementedError

    def _load_model(self) -> Any:
        if self.model_loader is None:
            raise NotImplementedError
        return self._load_with_cache_first(
            self.model_loader,
            self.model_id,
            "model",
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=self.model_torch_dtype,
        )

    def _extract_prompt(self, prompt: str) -> str:
        return (
            str(prompt or "")
            .replace("USER:", "")
            .replace("ASSISTANT:", "")
            .replace("<image>", "")
            .strip()
        )

    # This method exists as some classes will add behavior to `_extract_prompt`
    def _prepare_prompt(self, prompt: str) -> str:
        prepared = self._extract_prompt(prompt)
        return prepared or self.default_prompt

    def _sampling_generation_kwargs(self) -> dict[str, Any]:
        temperature = float(getattr(self, "temperature", 0.0))
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        return gen_kwargs


class AutoProcessorModelBase(HuggingFaceModelBase):
    input_artifact_attr = "processor"
    input_artifact_name = "processor"

    def _load_input_artifact(self) -> Any:
        return self._load_with_cache_first(
            AutoProcessor.from_pretrained,
            self.model_id,
            "processor",
        )


class AutoImageTextToTextModelBase(AutoProcessorModelBase):
    """Common wrapper for chat-template vision-language checkpoints."""

    default_model_id: str
    display_name = "Vision-language model"
    default_trust_remote_code = False

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.name = self.model_id.rsplit("/", 1)[-1].lower().replace("_", "-")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = (
            self.default_trust_remote_code
            if trust_remote_code is None
            else trust_remote_code
        )
        self._load_input_artifact_and_model()

    def _load_input_artifact(self) -> Any:
        return self._load_with_cache_first(
            AutoProcessor.from_pretrained,
            self.model_id,
            "processor",
            trust_remote_code=self.trust_remote_code,
        )

    def _load_model(self) -> Any:
        import transformers  # type: ignore

        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            raise ImportError(
                "AutoModelForImageTextToText is required for this multimodal checkpoint. "
                "Upgrade transformers to a version that supports it."
            )
        return self._load_with_cache_first(
            model_cls.from_pretrained,
            self.model_id,
            "model",
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=self.model_torch_dtype,
        )

    def predict(self, image: Image.Image, prompt: str) -> str:
        instruction = self._prepare_prompt(prompt)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model.generate(**inputs, **self._sampling_generation_kwargs())
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
