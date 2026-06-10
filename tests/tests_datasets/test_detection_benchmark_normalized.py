from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List

from PIL import Image

from benchmarks.detection import DetectionBenchmark
from dataset._base_dataset import BaseDataset


class _NormalizedBoxDataset(BaseDataset):
    def __init__(self) -> None:
        self.name = "normalized_detection"
        self.labels = ["plane"]
        self._rows = [
            {
                "image": Image.new("RGB", (1000, 500), "white"),
                "objects": [
                    {
                        "label": "plane",
                        "xmin": 0.1,
                        "ymin": 0.2,
                        "xmax": 0.7,
                        "ymax": 0.8,
                    }
                ],
            }
        ]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self._rows[:n]

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["image"]

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        return ["plane"]

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"label": "plane", "bbox": [0.1, 0.2, 0.6, 0.6]}]


class _EchoDetector:
    def predict(self, image: Image.Image, prompt: str) -> str:
        del image
        del prompt
        return "plane: [0.1, 0.2, 0.6, 0.6]"


class DetectionBenchmarkNormalizedTests(unittest.TestCase):
    def test_normalized_ground_truth_boxes_are_scaled_like_predictions(self) -> None:
        benchmark = DetectionBenchmark(dataset=_NormalizedBoxDataset(), name="normalized_detection")
        report = benchmark.run(
            model=_EchoDetector(),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["ground_truth_boxes"][0]["xyxy"], [100.0, 100.0, 700.0, 400.0])
        self.assertEqual(result["predicted_boxes"][0]["xyxy"], [100.0, 100.0, 700.0, 400.0])


if __name__ == "__main__":
    unittest.main()
