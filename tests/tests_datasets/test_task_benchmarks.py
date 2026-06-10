from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List

from PIL import Image

from benchmarks._base_benchmark import BaseBenchmark
from benchmarks import (
    Flickr30kEntitiesBenchmark,
    LVISBenchmark,
    MSCOCOBenchmark,
    OpenImagesV4DetectionBenchmark,
    UCF101Benchmark,
)
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


class _GroupedClassificationDataset(_ClassificationDataset):
    def __init__(self) -> None:
        self.name = "grouped-classification"
        self.labels = ["cat", "dog", "car"]
        self._rows = [
            {
                "id": f"{label}-{index}",
                "image": Image.new("RGB", (24, 24), color),
                "label": label,
            }
            for label, color in (("cat", "white"), ("dog", "gray"), ("car", "black"))
            for index in range(80)
        ]


class _SpacedClassificationDataset(_GroupedClassificationDataset):
    def __init__(self) -> None:
        super().__init__()
        self.spaced_sample_calls = 0

    def get_spaced_samples(self, n: int) -> List[Dict[str, Any]]:
        self.spaced_sample_calls += 1
        del n
        return [self._rows[0], self._rows[80], self._rows[160]]


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


class _MultiClassDetectionDataset(_DetectionDataset):
    def __init__(self) -> None:
        self.name = "multi_class_detection"
        self.labels = ["car", "person"]

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        return [item["label"] for item in row["annotations"]]

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(row["annotations"])


class TaskBenchmarkTests(unittest.TestCase):
    def test_classification_benchmark_scores_exact_label_match(self) -> None:
        benchmark = ClassificationBenchmark(dataset=_ClassificationDataset(), name="classification")
        report = benchmark.run(model=_PredictConstModel("cat"), n=1, label_sample_size=2, show_progress=False)

        self.assertEqual(report["num_samples"], 1)
        self.assertTrue(report["results"][0]["correct"])
        self.assertEqual(report["results"][0]["valid_labels"], ["cat"])

    def test_classification_prepare_prioritizes_distinct_labels_from_grouped_rows(self) -> None:
        benchmark = ClassificationBenchmark(
            dataset=_GroupedClassificationDataset(),
            name="grouped-classification",
        )

        rows, labels = benchmark.prepare(n=3, label_sample_size=3)

        self.assertEqual(labels, ["cat", "dog", "car"])
        self.assertEqual({row["label"] for row in rows}, {"cat", "dog", "car"})

    def test_classification_prepare_balances_repeats_after_labels_are_exhausted(self) -> None:
        benchmark = ClassificationBenchmark(
            dataset=_GroupedClassificationDataset(),
            name="grouped-classification",
        )

        rows, _ = benchmark.prepare(n=8, label_sample_size=3)
        counts = {label: sum(row["label"] == label for row in rows) for label in benchmark.dataset.labels}

        self.assertEqual(len(rows), 8)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_classification_prepare_uses_spaced_dataset_rows_when_available(self) -> None:
        dataset = _SpacedClassificationDataset()
        benchmark = ClassificationBenchmark(dataset=dataset, name="spaced-classification")

        rows, _ = benchmark.prepare(n=3, label_sample_size=3)

        self.assertEqual(dataset.spaced_sample_calls, 1)
        self.assertEqual({row["label"] for row in rows}, {"cat", "dog", "car"})

    def test_ucf101_benchmark_uses_one_representative_frame_per_clip(self) -> None:
        benchmark = UCF101Benchmark.__new__(UCF101Benchmark)
        BaseBenchmark.__init__(benchmark, dataset=_UCF101Dataset(), name="ucf101")
        benchmark.frames_per_clip = 3
        benchmark.search_limit = 16

        rows, labels = benchmark.prepare(n=2, label_sample_size=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(labels), ["basketball", "diving"])

        image = benchmark.get_image_for_row(rows[0])
        self.assertEqual(image.size, (16, 12))
        self.assertEqual(image.getpixel((0, 0)), (0, 128, 0))
        self.assertEqual(benchmark.get_valid_labels_for_row(rows[0]), ["basketball"])

    def test_detection_filters_unlabeled_prediction_to_single_allowed_label(self) -> None:
        benchmark = DetectionBenchmark(dataset=_DetectionDataset(), name="detection")
        filtered = benchmark._filter_prediction_boxes(
            row={},
            labels=["airplane"],
            predicted_boxes=[{"label": "", "xyxy": [0.0, 0.0, 1.0, 1.0]}],
        )
        self.assertEqual(filtered, [{"label": "airplane", "xyxy": [0.0, 0.0, 1.0, 1.0]}])

    def test_every_type_b_benchmark_targets_one_object_class_per_row(self) -> None:
        row_template = {
            "id": "shared-row",
            "image": Image.new("RGB", (100, 100), "white"),
            "annotations": [
                {"label": "car", "bbox": [10.0, 10.0, 20.0, 20.0]},
                {"label": "person", "bbox": [40.0, 10.0, 10.0, 30.0]},
                {"label": "car", "bbox": [70.0, 50.0, 20.0, 20.0]},
                {"label": "person", "bbox": [20.0, 60.0, 10.0, 30.0]},
            ],
        }

        for benchmark_cls in (
            Flickr30kEntitiesBenchmark,
            MSCOCOBenchmark,
            LVISBenchmark,
            OpenImagesV4DetectionBenchmark,
        ):
            with self.subTest(benchmark=benchmark_cls.__name__):
                row = dict(row_template)
                row["annotations"] = [dict(item) for item in row_template["annotations"]]
                benchmark = benchmark_cls.__new__(benchmark_cls)
                BaseBenchmark.__init__(
                    benchmark,
                    dataset=_MultiClassDetectionDataset(),
                    name=benchmark_cls.__name__,
                )

                prompt_labels = benchmark.get_prompt_labels_for_row(row, ["car", "person"])
                ground_truth = benchmark.get_ground_truth_boxes_for_row(row)
                prompt = benchmark.make_prompt(prompt_labels, row=row, image=row["image"])

                self.assertEqual(len(prompt_labels), 1)
                self.assertEqual(len(ground_truth), 2)
                self.assertEqual({box["label"] for box in ground_truth}, set(prompt_labels))
                self.assertIn(f"Target class: {prompt_labels[0]}", prompt)
                self.assertIn("Detect all visible instances of the target class", prompt)
                self.assertIn("[x_1, y_1, width_1, height_1]", prompt)
                self.assertIn("[x_2, y_2, width_2, height_2]", prompt)
                self.assertIn("top-left corner", prompt)
                self.assertIn("[0.5, 0.0, 0.5, 0.5]\n[0.0, 0.5, 0.5, 0.5]", prompt)
                self.assertIn("return exactly:\n\n[]", prompt)
                other_label = ({"car", "person"} - set(prompt_labels)).pop()
                self.assertNotIn(f"Target class: {other_label}", prompt)

    def test_base_benchmark_limits_prompt_labels_but_keeps_valid_label(self) -> None:
        benchmark = ClassificationBenchmark(dataset=_ClassificationDataset(), name="classification")
        row = {"image": Image.new("RGB", (24, 24), "white"), "label": "cat"}
        labels = [f"label-{idx}" for idx in range(20)] + ["cat"]
        prompt_labels = benchmark.get_prompt_labels_for_row(row, labels)

        self.assertLessEqual(len(prompt_labels), benchmark.MAX_PROMPT_LABELS)
        self.assertIn("cat", prompt_labels)


if __name__ == "__main__":
    unittest.main()
