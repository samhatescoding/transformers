from __future__ import annotations

import re
import unittest

from benchmarks.multiple_choice._multiple_choice import MultipleChoiceBenchmark
from benchmarks.visual_qa._visual_qa import VisualQABenchmark


class _NormalizeDataset:
    @staticmethod
    def normalize_text(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


class ChoiceParsingTests(unittest.TestCase):
    def test_visual_qa_parses_verbose_choice_letter(self) -> None:
        benchmark = VisualQABenchmark.__new__(VisualQABenchmark)
        benchmark.dataset = _NormalizeDataset()

        selected = benchmark._parse_choice(
            "Based on the image, the correct choice is **A**.",
            ["carnival ride", "playground", "school bus", "bicycle"],
        )

        self.assertEqual(selected, "carnival ride")

    def test_visual_qa_parses_uniquely_mentioned_choice_text(self) -> None:
        benchmark = VisualQABenchmark.__new__(VisualQABenchmark)
        benchmark.dataset = _NormalizeDataset()

        selected = benchmark._parse_choice(
            "The image shows a carnival ride with children on it.",
            ["carnival ride", "playground", "school bus", "bicycle"],
        )

        self.assertEqual(selected, "carnival ride")

    def test_multiple_choice_parses_verbose_choice_letter(self) -> None:
        benchmark = MultipleChoiceBenchmark.__new__(MultipleChoiceBenchmark)
        benchmark.dataset = _NormalizeDataset()

        selected = benchmark._parse_choice(
            "Therefore, the answer is (C).",
            ["first", "second", "third", "fourth"],
        )

        self.assertEqual(selected, "third")

    def test_ambiguous_choice_text_falls_back_to_raw_prediction(self) -> None:
        benchmark = MultipleChoiceBenchmark.__new__(MultipleChoiceBenchmark)
        benchmark.dataset = _NormalizeDataset()
        prediction = "Both red apple and green apple are visible."

        selected = benchmark._parse_choice(
            prediction,
            ["red apple", "green apple", "banana", "orange"],
        )

        self.assertEqual(selected, prediction)


if __name__ == "__main__":
    unittest.main()
