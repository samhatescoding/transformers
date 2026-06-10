from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List

from PIL import Image

from benchmarks._base_benchmark import BaseBenchmark
from benchmarks import Flickr30kBenchmark
from dataset._base_dataset import BaseDataset


class _StubFlickr30kDataset(BaseDataset):
    def __init__(self) -> None:
        self.name = "flickr30k"
        self.labels = []
        self._rows = [
            {
                "image": Image.new("RGB", (32, 24), (255, 255, 255)),
                "caption": [
                    "A dog runs through the grass.",
                    "A brown dog is running in a field.",
                    "A dog sprints across a grassy area.",
                    "A dog runs outdoors on grass.",
                    "A fast dog is running in the grass.",
                ],
            }
        ]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self._rows[:n]

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["image"]

    def get_captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        return list(row.get("caption", []))


class _CaptionModel:
    def predict(self, image: Image.Image, prompt: str) -> str:
        del image
        del prompt
        return "A brown dog is running in a field."


class _StubFlickr30kBenchmark(Flickr30kBenchmark):
    def __init__(self, dataset: _StubFlickr30kDataset) -> None:
        BaseBenchmark.__init__(self, dataset=dataset, name="flickr30k")
        self.bleu_threshold = 0.25


class Flickr30kCaptionBenchmarkTests(unittest.TestCase):
    def test_caption_benchmark_scores_prediction_against_reference_captions(self) -> None:
        benchmark = _StubFlickr30kBenchmark(dataset=_StubFlickr30kDataset())
        report = benchmark.run(
            model=_CaptionModel(),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        self.assertEqual(report["benchmark"], "flickr30k")
        self.assertEqual(report["dataset"], "flickr30k")
        self.assertEqual(report["num_samples"], 1)
        self.assertEqual(report["num_candidate_labels"], 0)

        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertEqual(result["prediction"], "A brown dog is running in a field.")
        self.assertEqual(len(result["valid_labels"]), 5)
        self.assertAlmostEqual(result["bleu"], 1.0, places=6)
        self.assertEqual(len(result["reference_captions"]), 5)


if __name__ == "__main__":
    unittest.main()
