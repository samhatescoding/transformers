from __future__ import annotations

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ._base_model import BaseModel


class MiniCPMV26(BaseModel):
    def __init__(
        self,
        model_id: str = "openbmb/MiniCPM-V-2_6",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.name = self._name_from_model_id(model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print("Loading MiniCPM tokenizer...")
        self.tokenizer = self._load_with_local_fallback(
            AutoTokenizer.from_pretrained,
            model_id,
            "tokenizer",
            trust_remote_code=True,
        )

        print("Loading MiniCPM model...")
        load_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        self.model = self._load_with_local_fallback(
            AutoModel.from_pretrained,
            model_id,
            "model",
            **load_kwargs,
        ).eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.to("cpu")

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        if model_id == "openbmb/MiniCPM-V-2_6":
            return "minicpm-v-2_6"
        return model_id.rsplit("/", 1)[-1].lower().replace(".", "").replace("_", "-")

    def _load_with_local_fallback(self, loader, model_id: str, artifact_name: str, **kwargs):
        try:
            return loader(model_id, **kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to load MiniCPM {artifact_name} from hub: {exc}")
            print(f"[INFO] Retrying MiniCPM {artifact_name} load from local cache only.")
            try:
                return loader(model_id, local_files_only=True, **kwargs)
            except Exception as local_exc:
                print(f"[WARN] Repo-id cache load for MiniCPM {artifact_name} failed: {local_exc}")
                snapshot_path = snapshot_download(model_id, local_files_only=True)
                print(f"[INFO] Retrying MiniCPM {artifact_name} from cached snapshot: {snapshot_path}")
                return loader(snapshot_path, local_files_only=True, **kwargs)

    def predict(self, image: Image.Image, prompt: str) -> str:
        message = {
            "role": "user",
            "content": [image.convert("RGB"), self._adapt_prompt(prompt)],
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

    def _adapt_prompt(self, prompt: str) -> str:
        return (
            str(prompt or "")
            .replace("USER:", "")
            .replace("ASSISTANT:", "")
            .replace("<image>", "")
            .strip()
            or "Describe the image."
        )
