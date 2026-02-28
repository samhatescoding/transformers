# models/gpt4.py

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

from PIL import Image

from .base import BaseModel


class GPT4(BaseModel):
    """
    GPT-4-class vision model wrapper using OpenAI's Responses API.

    Requirements:
      pip install openai

    Auth:
      export OPENAI_API_KEY="..."
      (on Windows PowerShell) setx OPENAI_API_KEY "..."
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",        # vision-capable default
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        self.name = "gpt4"
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Import here so the rest of your project can run without OpenAI installed
        from openai import OpenAI  # type: ignore

        # If api_key is None, OpenAI() will use OPENAI_API_KEY env var
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _pil_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
        """
        Convert a PIL image to a base64 data URL suitable for OpenAI image input.
        """
        buf = BytesIO()
        # JPEG is smaller; PNG is lossless. Pick one.
        image.convert("RGB").save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _strip_llava_image_token(prompt: str) -> str:
        # Your prompts may contain "USER: <image>" from LLaVA formatting.
        # GPT doesn't need that; remove it if present.
        return prompt.replace("<image>", "").strip()

    def predict(self, image: Image.Image, prompt: str) -> str:
        prompt = self._strip_llava_image_token(prompt)
        image_url = self._pil_to_data_url(image, fmt="JPEG")

        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            max_output_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        # The Python SDK provides a convenience property output_text
        return (response.output_text or "").strip()