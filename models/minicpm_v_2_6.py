from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ._hf_model import HuggingFaceModelBase


class MiniCPMV26(HuggingFaceModelBase):
    display_name = "MiniCPM"
    input_artifact_attr = "tokenizer"
    input_artifact_name = "tokenizer"
    default_model_id = "openbmb/MiniCPM-V-2_6"

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.name = self._name_from_model_id(self.model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._load_input_artifact_and_model()
        self.model = self.model.eval()
        self.model = self.model.cuda()

    def _load_input_artifact(self):
        return self._load_with_cache_first(
            AutoTokenizer.from_pretrained,
            self.model_id,
            "tokenizer",
            trust_remote_code=True,
        )

    def _load_model(self):
        return self._load_with_cache_first(
            AutoModel.from_pretrained,
            self.model_id,
            "model",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        known_names = {
            "openbmb/MiniCPM-V-2_6": "minicpm-v-2.6",
            "openbmb/MiniCPM-o-2_6": "minicpm-o-2.6",
        }
        if model_id in known_names:
            return known_names[model_id]
        return model_id.rsplit("/", 1)[-1].lower().replace("_", "-")

    def predict(self, image: Image.Image, prompt: str) -> str:
        message = {
            "role": "user",
            "content": [image.convert("RGB"), self._prepare_prompt(prompt)],
        }
        kwargs = {
            "image": None,
            "msgs": [message],
            "tokenizer": self.tokenizer,
            "sampling": self.temperature > 0,
            "stream": False,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        response = self.model.chat(**kwargs)
        return str(response).strip()
