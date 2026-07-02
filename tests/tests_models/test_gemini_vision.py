from __future__ import annotations

import sys
import types as module_types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from models._gemini_vision import GeminiVisionModel


class _TestGeminiModel(GeminiVisionModel):
    default_model_id = "test-gemini"


class _FakeThinkingConfig:
    def __init__(self, thinking_budget=None) -> None:
        self.thinking_budget = thinking_budget


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FallbackGenerateContentConfig:
    def __init__(self, **kwargs) -> None:
        if "thinking_config" in kwargs:
            raise TypeError("thinking_config is unsupported")
        self.kwargs = kwargs


class GeminiVisionModelTests(unittest.TestCase):
    def _patch_google_genai(self, generate_config_cls=_FakeGenerateContentConfig):
        google_module = module_types.ModuleType("google")
        genai_module = module_types.ModuleType("google.genai")
        types_module = module_types.ModuleType("google.genai.types")
        types_module.GenerateContentConfig = generate_config_cls
        types_module.ThinkingConfig = _FakeThinkingConfig
        genai_module.types = types_module
        genai_module.Client = lambda api_key=None: SimpleNamespace(api_key=api_key)
        google_module.genai = genai_module
        return patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "google.genai.types": types_module,
            },
        )

    def test_raises_minimum_visible_output_budget_and_disables_thinking(self) -> None:
        with self._patch_google_genai():
            model = _TestGeminiModel(max_new_tokens=16, temperature=0.2, api_key="test-key")
            requests = []
            model.client = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **kwargs: requests.append(kwargs) or SimpleNamespace(text=" answer ")
                )
            )

            answer = model.predict(Image.new("RGB", (2, 2), "white"), "USER: <image>\nQuestion?")

        self.assertEqual(answer, "answer")
        self.assertEqual(requests[0]["model"], "test-gemini")
        self.assertEqual(
            requests[0]["contents"][0],
            "USER: \nQuestion?\n\n"
            "Return only the requested final answer in the requested format. "
            "Do not include explanations, markdown, or analysis.",
        )
        config = requests[0]["config"]
        self.assertEqual(config.kwargs["max_output_tokens"], 128)
        self.assertEqual(config.kwargs["temperature"], 0.2)
        self.assertEqual(config.kwargs["thinking_config"].thinking_budget, 0)

    def test_generation_config_falls_back_when_thinking_is_unsupported(self) -> None:
        with self._patch_google_genai(generate_config_cls=_FallbackGenerateContentConfig):
            model = _TestGeminiModel(max_new_tokens=128)
            config = model._make_generation_config()

        self.assertEqual(config.kwargs["max_output_tokens"], 128)
        self.assertNotIn("thinking_config", config.kwargs)

    def test_extracts_candidate_parts_when_response_text_is_empty(self) -> None:
        response = SimpleNamespace(
            text="",
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="part "),
                            SimpleNamespace(text="text"),
                        ]
                    )
                )
            ],
        )

        self.assertEqual(_TestGeminiModel._extract_text(response), "part text")


if __name__ == "__main__":
    unittest.main()
