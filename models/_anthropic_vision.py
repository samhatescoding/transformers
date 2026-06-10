from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

from PIL import Image

from ._base_model import BaseModel


class AnthropicVisionModel(BaseModel):
    """Image-to-text model accessed through Anthropic's Messages API."""

    default_model_id: str

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

        from anthropic import Anthropic  # type: ignore

        self.client = Anthropic(api_key=api_key)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, image: Image.Image, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": self._encode_image(image),
                            },
                        },
                        {
                            "type": "text",
                            "text": str(prompt or "").replace("<image>", "").strip(),
                        },
                    ],
                }
            ],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
