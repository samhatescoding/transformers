from __future__ import annotations

from typing import Optional

from PIL import Image

from ._base_model import BaseModel


class GeminiVisionModel(BaseModel):
    """Image-to-text model accessed through the Google Gen AI SDK."""

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

        from google import genai  # type: ignore

        self.client = genai.Client(api_key=api_key)

    def predict(self, image: Image.Image, prompt: str) -> str:
        from google.genai import types  # type: ignore

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[str(prompt or "").replace("<image>", "").strip(), image],
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_new_tokens,
                temperature=self.temperature,
            ),
        )
        return (response.text or "").strip()
