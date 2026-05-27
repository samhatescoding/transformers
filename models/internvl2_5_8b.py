from __future__ import annotations

from itertools import product
from math import inf

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ._hf_model import HuggingFaceModelBase

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class InternVL25(HuggingFaceModelBase):
    display_name = "InternVL"
    input_artifact_attr = "tokenizer"
    input_artifact_name = "tokenizer"
    default_model_id = "OpenGVLab/InternVL2_5-8B"

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        image_size: int = 448,
        max_num_tiles: int = 12,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.name = self._name_from_model_id(self.model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.image_size = int(image_size)
        self.max_num_tiles = int(max_num_tiles)

        self._load_input_artifact_and_model()
        self.model = self.model.eval()
        self.model = self.model.cuda()

    def _load_input_artifact(self):
        return self._load_with_cache_first(
            AutoTokenizer.from_pretrained,
            self.model_id,
            "tokenizer",
            trust_remote_code=True,
            use_fast=False,
        )

    def _load_model(self):
        return self._load_with_cache_first(
            AutoModel.from_pretrained,
            self.model_id,
            "model",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=True,
        )

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        if model_id == "OpenGVLab/InternVL2_5-8B":
            return "internvl2.5-8b"
        return model_id.rsplit("/", 1)[-1].lower().replace("_", "-")

    def predict(self, image: Image.Image, prompt: str) -> str:
        pixel_values = self._load_image(image).to(
            device=next(self.model.parameters()).device,
            dtype=torch.bfloat16,
        )
        question = self._prepare_prompt(prompt)
        generation_config = self._sampling_generation_kwargs()
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()

    def _prepare_prompt(self, prompt: str) -> str:
        instruction = self._extract_prompt(prompt)
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
