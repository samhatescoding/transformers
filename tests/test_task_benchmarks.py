from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List

from PIL import Image

from benchmarks._base_benchmark import BaseBenchmark
from benchmarks import UCF101Benchmark
from benchmarks.classification import ClassificationBenchmark
from benchmarks.detection import DetectionBenchmark


class _ClassificationDataset:
    def __init__(self) -> None:
        self.name = "classification"
        self.labels = ["cat", "dog", "car"]
        self._rows = [
            {"image": Image.new("RGB", (24, 24), "white"), "label": "cat"},
            {"image": Image.new("RGB", (24, 24), "gray"), "label": "dog"},
        ]

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self._rows[:n]

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        return [row["label"]]

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["image"]

    def normalize_text(self, text: str) -> str:
        return str(text).strip().lower()


class _UCF101Dataset:
    def __init__(self) -> None:
        self.name = "ucf101"
        self.labels = ["basketball", "diving"]
        self._rows = [
            {"clip_id": "clip-1", "frame": 0, "image": Image.new("RGB", (16, 12), "red"), "label": 0},
            {"clip_id": "clip-1", "frame": 1, "image": Image.new("RGB", (16, 12), "green"), "label": 0},
            {"clip_id": "clip-1", "frame": 2, "image": Image.new("RGB", (16, 12), "blue"), "label": 0},
            {"clip_id": "clip-2", "frame": 0, "image": Image.new("RGB", (16, 12), "yellow"), "label": 1},
            {"clip_id": "clip-2", "frame": 1, "image": Image.new("RGB", (16, 12), "orange"), "label": 1},
            {"clip_id": "clip-2", "frame": 2, "image": Image.new("RGB", (16, 12), "purple"), "label": 1},
        ]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["image"]

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        return [self.labels[int(row["label"])]]

    def normalize_text(self, text: str) -> str:
        return str(text).strip().lower()


class _PredictConstModel:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, image: Image.Image, prompt: str) -> str:
        del image
        del prompt
        return self.prediction


class _DetectionDataset:
    def __init__(self) -> None:
        self.name = "detection"
        self.labels = ["airplane"]

    def normalize_text(self, text: str) -> str:
        return str(text).strip().lower()


class TaskBenchmarkTests(unittest.TestCase):
    def test_classification_benchmark_scores_exact_label_match(self) -> None:
        benchmark = ClassificationBenchmark(dataset=_ClassificationDataset(), name="classification")
        report = benchmark.run(model=_PredictConstModel("cat"), n=1, label_sample_size=2, show_progress=False)

        self.assertEqual(report["num_samples"], 1)
        self.assertTrue(report["results"][0]["correct"])
        self.assertEqual(report["results"][0]["valid_labels"], ["cat"])

    def test_ucf101_benchmark_groups_clip_frames(self) -> None:
        benchmark = UCF101Benchmark.__new__(UCF101Benchmark)
        BaseBenchmark.__init__(benchmark, dataset=_UCF101Dataset(), name="ucf101")
        benchmark.frames_per_clip = 3
        benchmark.search_limit = 16

        rows, labels = benchmark.prepare(n=2, label_sample_size=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(labels), ["basketball", "diving"])

        image = benchmark.get_image_for_row(rows[0])
        self.assertEqual(image.size, (48, 36))
        self.assertEqual(benchmark.get_valid_labels_for_row(rows[0]), ["basketball"])

    def test_detection_filters_unlabeled_prediction_to_single_allowed_label(self) -> None:
        benchmark = DetectionBenchmark(dataset=_DetectionDataset(), name="detection")
        filtered = benchmark._filter_prediction_boxes(
            row={},
            labels=["airplane"],
            predicted_boxes=[{"label": "", "xyxy": [0.0, 0.0, 1.0, 1.0]}],
        )
        self.assertEqual(filtered, [{"label": "airplane", "xyxy": [0.0, 0.0, 1.0, 1.0]}])

    def test_base_benchmark_limits_prompt_labels_but_keeps_valid_label(self) -> None:
        benchmark = ClassificationBenchmark(dataset=_ClassificationDataset(), name="classification")
        row = {"image": Image.new("RGB", (24, 24), "white"), "label": "cat"}
        labels = [f"label-{idx}" for idx in range(20)] + ["cat"]
        prompt_labels = benchmark.get_prompt_labels_for_row(row, labels)

        self.assertLessEqual(len(prompt_labels), benchmark.MAX_PROMPT_LABELS)
        self.assertIn("cat", prompt_labels)


if __name__ == "__main__":
    unittest.main()
