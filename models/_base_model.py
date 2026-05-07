from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class BaseModel(ABC):
    name: str
    tokens: int

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError

    def get_tokenizer(self) -> Any | None:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        processor = getattr(self, "processor", None)
        if processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
        return None

    def count_text_tokens(self, text: str) -> int | None:
        tokenizer = self.get_tokenizer()
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer(
                text or "",
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except Exception:
            return None

        input_ids = encoded.get("input_ids")
        if input_ids is None:
            return None
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
