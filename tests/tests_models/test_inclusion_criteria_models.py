from __future__ import annotations

import re
import unittest
from pathlib import Path

import models
from models import BaseModel


MODEL_CATALOG = {
    "gpt-5.5": ("GPT55", "gpt-5.5"),
    "gpt-5.4": ("GPT54", "gpt-5.4"),
    "o3": ("O3", "o3"),
    "o4-mini": ("O4Mini", "o4-mini"),
    "claude-opus-4-8": ("ClaudeOpus48", "claude-opus-4-8"),
    "claude-opus-4-7": ("ClaudeOpus47", "claude-opus-4-7"),
    "claude-opus-4-6": ("ClaudeOpus46", "claude-opus-4-6"),
    "claude-sonnet-4-6": ("ClaudeSonnet46", "claude-sonnet-4-6"),
    "claude-sonnet-4-5": ("ClaudeSonnet45", "claude-sonnet-4-5"),
    "gemini-3.1-pro-preview": ("Gemini31ProPreview", "gemini-3.1-pro-preview"),
    "gemini-3.5-flash": ("Gemini35Flash", "gemini-3.5-flash"),
    "qwen3.7-plus-preview": ("Qwen37PlusPreview", "qwen3.7-plus-preview"),
    "qwen3.6-plus": ("Qwen36Plus", "qwen3.6-plus"),
    "qwen3.6-27b": ("Qwen36_27B", "Qwen/Qwen3.6-27B"),
    "qwen3.5-397b-a17b": ("Qwen35_397BA17B", "Qwen/Qwen3.5-397B-A17B"),
    "qwen3-vl-235b-a22b": ("Qwen3VL235BA22B", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
    "kimi-k2.6": ("KimiK26", "moonshotai/Kimi-K2.6"),
    "kimi-k2.5-thinking": ("KimiK25Thinking", "moonshotai/Kimi-K2.5"),
    "gemma-4-31b": ("Gemma4_31B", "google/gemma-4-31B-it"),
    "gemma-4-26b-a4b": ("Gemma4_26BA4B", "google/gemma-4-26B-A4B-it"),
    "mimo-v2.5": ("MiMoV25", "XiaomiMiMo/MiMo-V2.5"),
    "mimo-v2.5-pro": ("MiMoV25Pro", "XiaomiMiMo/MiMo-V2.5-Pro"),
    "mistral-medium-3.5": ("MistralMedium35", "mistralai/Mistral-Medium-3.5-128B"),
    "mistral-small-4": ("MistralSmall4", "mistralai/Mistral-Small-4-119B-2603"),
    "dots.vlm1": ("DotsVLM1", "rednote-hilab/dots.vlm1.inst"),
    "dots.ocr": ("DotsOCR", "rednote-hilab/dots.ocr"),
    "glm-4.5v-thinking": ("GLM45VThinking", "zai-org/GLM-4.5V"),
    "glm-4.1v-thinking": ("GLM41VThinking", "zai-org/GLM-4.1V-Thinking"),
    "skywork-r1v3-38b": ("SkyworkR1V3_38B", "Skywork/Skywork-R1V3-38B"),
    "skywork-r1v2-38b": ("SkyworkR1V2_38B", "Skywork/Skywork-R1V2-38B"),
}


class InclusionCriteriaModelTests(unittest.TestCase):
    def test_every_selected_model_has_a_public_wrapper(self) -> None:
        tex_path = Path(__file__).resolve().parents[2] / "docs" / "tex" / "Inclusion Criteria.tex"
        text = tex_path.read_text(encoding="utf-8")
        selection = text.split(r"\section{The Procedure}", 1)[0]
        selected_names = set(re.findall(r"\\item \\texttt\{([^}]+)\}", selection))

        self.assertEqual(selected_names, set(MODEL_CATALOG))
        for selected_name, (public_name, model_id) in MODEL_CATALOG.items():
            model_class = getattr(models, public_name)
            self.assertTrue(issubclass(model_class, BaseModel), selected_name)
            self.assertIn(public_name, models.__all__)
            self.assertEqual(model_class.default_model_id, model_id, selected_name)

    def test_custom_checkpoint_wrappers_enable_remote_code_by_default(self) -> None:
        for public_name in (
            "KimiK26",
            "KimiK25Thinking",
            "MiMoV25",
            "MiMoV25Pro",
            "MistralMedium35",
            "MistralSmall4",
            "DotsVLM1",
            "DotsOCR",
            "GLM45VThinking",
            "GLM41VThinking",
            "SkyworkR1V3_38B",
            "SkyworkR1V2_38B",
        ):
            self.assertTrue(getattr(models, public_name).default_trust_remote_code, public_name)


if __name__ == "__main__":
    unittest.main()
