from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Optional

from PIL import Image

from ._base_model import BaseModel


class OpenAIResponsesVisionModel(BaseModel):
    """Image-to-text model accessed through OpenAI's Responses API."""

    default_model_id: str
    supports_temperature = True
    reasoning_effort: str | None = None
    min_output_tokens = 16

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.name = self.model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        from openai import OpenAI  # type: ignore

        resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if resolved_api_key is not None:
            resolved_api_key = resolved_api_key.strip()
        self.client = OpenAI(api_key=resolved_api_key)

    @staticmethod
    def _pil_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
        buf = BytesIO()
        image.convert("RGB").save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _strip_llava_image_token(prompt: str) -> str:
        return str(prompt or "").replace("<image>", "").strip()

    @classmethod
    def _prepare_prompt(cls, prompt: str) -> str:
        return cls._strip_llava_image_token(prompt)

    def predict(self, image: Image.Image, prompt: str) -> str:
        request = {
            "model": self.model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self._prepare_prompt(prompt)},
                        {"type": "input_image", "image_url": self._pil_to_data_url(image)},
                    ],
                }
            ],
            "max_output_tokens": max(self.max_new_tokens, self.min_output_tokens),
        }
        if self.supports_temperature:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**request)
        return (response.output_text or "").strip()


class GPT5VisionModel(OpenAIResponsesVisionModel):
    supports_temperature = False
