from __future__ import annotations

from itertools import product
from math import inf

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ._base_model import BaseModel

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class InternVL25(BaseModel):
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-8B",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        image_size: int = 448,
        max_num_tiles: int = 12,
    ) -> None:
        self.model_id = model_id
        self.name = self._name_from_model_id(model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.image_size = int(image_size)
        self.max_num_tiles = int(max_num_tiles)

        print("Loading InternVL tokenizer...")
        self.tokenizer = self._load_with_local_fallback(
            AutoTokenizer.from_pretrained,
            model_id,
            "tokenizer",
            trust_remote_code=True,
            use_fast=False,
        )

        print("Loading InternVL model...")
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": bool(torch.cuda.is_available()),
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            load_kwargs["use_flash_attn"] = True
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
        if model_id == "OpenGVLab/InternVL2_5-8B":
            return "internvl25-8b"
        return model_id.rsplit("/", 1)[-1].lower().replace(".", "").replace("_", "-")

    def _load_with_local_fallback(self, loader, model_id: str, artifact_name: str, **kwargs):
        try:
            return loader(model_id, **kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to load InternVL {artifact_name} from hub: {exc}")
            print(f"[INFO] Retrying InternVL {artifact_name} load from local cache only.")
            try:
                return loader(model_id, local_files_only=True, **kwargs)
            except Exception as local_exc:
                print(f"[WARN] Repo-id cache load for InternVL {artifact_name} failed: {local_exc}")
                snapshot_path = snapshot_download(model_id, local_files_only=True)
                print(f"[INFO] Retrying InternVL {artifact_name} from cached snapshot: {snapshot_path}")
                return loader(snapshot_path, local_files_only=True, **kwargs)

    def predict(self, image: Image.Image, prompt: str) -> str:
        pixel_values = self._load_image(image).to(
            device=next(self.model.parameters()).device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        question = self._adapt_prompt(prompt)
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_config["temperature"] = self.temperature
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()

    def _adapt_prompt(self, prompt: str) -> str:
        instruction = (
            str(prompt or "")
            .replace("USER:", "")
            .replace("ASSISTANT:", "")
            .strip()
        )
        if "<image>" not in instruction:
            instruction = f"<image>\n{instruction}"
        return instruction.strip() or "<image>\nDescribe the image."

    def _load_image(self, image: Image.Image) -> torch.Tensor:
        tiles = self._dynamic_preprocess(image.convert("RGB"))
        return torch.stack([self._transform(tile) for tile in tiles], dim=0)

    def _transform(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        storage = torch.ByteStorage.from_buffer(resized.tobytes())
        tensor = torch.ByteTensor(storage).view(resized.height, resized.width, 3).permute(2, 0, 1).float() / 255.0
        return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD

    def _dynamic_preprocess(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        aspect_ratio = width / height
        target_ratio = self._find_closest_aspect_ratio(aspect_ratio)
        target_width = self.image_size * target_ratio[0]
        target_height = self.image_size * target_ratio[1]
        blocks = target_ratio[0] * target_ratio[1]

        resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        processed: list[Image.Image] = []
        tiles_per_row = target_width // self.image_size
        for idx in range(blocks):
            box = (
                (idx % tiles_per_row) * self.image_size,
                (idx // tiles_per_row) * self.image_size,
                ((idx % tiles_per_row) + 1) * self.image_size,
                ((idx // tiles_per_row) + 1) * self.image_size,
            )
            processed.append(resized.crop(box))
        if len(processed) != 1:
            processed.append(image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC))
        return processed

    def _find_closest_aspect_ratio(self, aspect_ratio: float) -> tuple[int, int]:
        target_ratios = {
            ratio
            for n in range(1, self.max_num_tiles + 1)
            for ratio in product(range(1, n + 1), repeat=2)
            if 1 <= ratio[0] * ratio[1] <= self.max_num_tiles
        }
        best_ratio = (1, 1)
        best_diff = inf
        for ratio in target_ratios:
            candidate = ratio[0] / ratio[1]
            diff = abs(aspect_ratio - candidate)
            if diff < best_diff:
                best_diff = diff
                best_ratio = ratio
        return best_ratio
