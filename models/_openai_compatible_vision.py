from __future__ import annotations

import os
from typing import Optional

from PIL import Image

from ._openai_vision import OpenAIResponsesVisionModel


class OpenAICompatibleVisionModel(OpenAIResponsesVisionModel):
    """Vision adapter for providers exposing an OpenAI-compatible API."""

    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: str | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.name = self.model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        from openai import OpenAI  # type: ignore

        resolved_key = api_key or (
            os.getenv(self.api_key_env) if self.api_key_env else None
        )
        resolved_url = base_url or (
            os.getenv(self.base_url_env) if self.base_url_env else None
        ) or self.base_url
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    def predict(self, image: Image.Image, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._strip_llava_image_token(prompt),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._pil_to_data_url(image)},
                        },
                    ],
                }
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()
