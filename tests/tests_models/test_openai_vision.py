from __future__ import annotations

import unittest
from unittest.mock import patch

from models._openai_vision import OpenAIResponsesVisionModel


class _TestOpenAIModel(OpenAIResponsesVisionModel):
    default_model_id = "test-model"


class OpenAIResponsesVisionModelTests(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_strips_whitespace_from_environment_api_key(self, mock_openai) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key\r\n"}):
            _TestOpenAIModel()

        mock_openai.assert_called_once_with(api_key="test-key")

    @patch("openai.OpenAI")
    def test_strips_whitespace_from_explicit_api_key(self, mock_openai) -> None:
        _TestOpenAIModel(api_key="  test-key\n")

        mock_openai.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
