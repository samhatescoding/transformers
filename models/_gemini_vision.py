from __future__ import annotations

from typing import Optional

from PIL import Image

from ._base_model import BaseModel


class GeminiVisionModel(BaseModel):
    """Image-to-text model accessed through the Google Gen AI SDK."""

    default_model_id: str
    min_output_tokens = 128

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

    def _make_generation_config(self):
        from google.genai import types  # type: ignore

        config_kwargs = {
            "max_output_tokens": max(self.max_new_tokens, self.min_output_tokens),
            "temperature": self.temperature,
        }
        thinking_config_cls = getattr(types, "ThinkingConfig", None)
        if thinking_config_cls is not None:
            config_kwargs["thinking_config"] = thinking_config_cls(thinking_budget=0)
        try:
            return types.GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("thinking_config", None)
            return types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _extract_text(response) -> str:
        try:
            text = response.text
        except Exception:
            text = None
        if text:
            return str(text).strip()

        chunks: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(str(part_text))
        return "".join(chunks).strip()

    @staticmethod
    def _prepare_prompt(prompt: str) -> str:
        text = str(prompt or "").replace("<image>", "").strip()
        return (
            text
            + "\n\nReturn only the requested final answer in the requested format. "
            "Do not include explanations, markdown, or analysis."
        )

    def predict(self, image: Image.Image, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[self._prepare_prompt(prompt), image],
            config=self._make_generation_config(),
        )
        return self._extract_text(response)
