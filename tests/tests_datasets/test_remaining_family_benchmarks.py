from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List

from PIL import Image

from benchmarks._base_benchmark import BaseBenchmark
from benchmarks import (
    Flickr30kEntitiesBenchmark,
    MSCOCOCaptionBenchmark,
    OpenImagesV4DetectionBenchmark,
)
from dataset._base_dataset import BaseDataset


class _CaptionStubDataset(BaseDataset):
    def __init__(self) -> None:
        self.name = "mscoco_caption"
        self.labels = []
        self._rows = [
            {
                "image": Image.new("RGB", (32, 24), (255, 255, 255)),
                "captions": [
                    "A small dog runs across a field.",
                    "A dog is running in the grass.",
                    "A brown dog sprints outdoors.",
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

    def get_labels_img(self, row) -> List[str]:
        del row
        return []

    def get_captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        return list(row["captions"])


class _DetectionStubDataset(BaseDataset):
    def __init__(self, name: str, row: Dict[str, Any], labels: List[str]) -> None:
        self.name = name
        self.labels = list(labels)
        self._rows = [row]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self._rows[:n]

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["image"]

    def get_labels_img(self, row) -> List[str]:
        return [item["label"] for item in row.get("annotations", [])]

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(row.get("annotations", []))

    def normalize_text(self, text: str) -> str:
        text = str(text).strip().lower()
        for ch in [".", ",", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}"]:
            text = text.replace(ch, "")
        return " ".join(text.split())


class _CaptionModel:
    def predict(self, image: Image.Image, prompt: str) -> str:
        del image
        del prompt
        return "A dog is running in the grass."


class _DetectionModel:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, image: Image.Image, prompt: str) -> str:
        del image
        del prompt
        return self.prediction


class _StubMSCOCOCaptionBenchmark(MSCOCOCaptionBenchmark):
    def __init__(self, dataset: _CaptionStubDataset) -> None:
        BaseBenchmark.__init__(self, dataset=dataset, name="mscoco_caption")
        self.bleu_threshold = 0.25


class _StubOpenImagesV4DetectionBenchmark(OpenImagesV4DetectionBenchmark):
    def __init__(self, dataset: _DetectionStubDataset) -> None:
        BaseBenchmark.__init__(self, dataset=dataset, name="openimages_v4_detection")


class _StubFlickr30kEntitiesBenchmark(Flickr30kEntitiesBenchmark):
    def __init__(self, dataset: _DetectionStubDataset) -> None:
        BaseBenchmark.__init__(self, dataset=dataset, name="flickr30k_entities")


class RemainingFamilyBenchmarkTests(unittest.TestCase):
    def test_mscoco_caption_benchmark_scores_against_reference_captions(self) -> None:
        benchmark = _StubMSCOCOCaptionBenchmark(dataset=_CaptionStubDataset())
        report = benchmark.run(model=_CaptionModel(), n=1, label_sample_size=1, show_progress=False)

        self.assertEqual(report["benchmark"], "mscoco_caption")
        self.assertEqual(report["dataset"], "mscoco_caption")
        self.assertEqual(report["num_candidate_labels"], 0)
        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["bleu"], 1.0, places=6)
        self.assertEqual(len(result["reference_captions"]), 3)

    def test_openimages_detection_benchmark_scores_box_predictions(self) -> None:
        row = {
            "image": Image.new("RGB", (100, 100), "white"),
            "annotations": [{"label": "car", "bbox": [10.0, 20.0, 30.0, 40.0]}],
        }
        dataset = _DetectionStubDataset("openimages_v4_detection", row=row, labels=["car"])
        benchmark = _StubOpenImagesV4DetectionBenchmark(dataset=dataset)

        report = benchmark.run(
            model=_DetectionModel("car: [0.1, 0.2, 0.3, 0.4]"),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["f1"], 1.0, places=6)
        self.assertEqual(result["valid_labels"], ["car"])

    def test_flickr30k_entities_benchmark_scores_phrase_grounding_boxes(self) -> None:
        row = {
            "image": Image.new("RGB", (100, 80), "white"),
            "annotations": [{"label": "yellow helmet", "bbox": [25.0, 10.0, 20.0, 20.0]}],
        }
        dataset = _DetectionStubDataset("flickr30k_entities", row=row, labels=["yellow helmet"])
        benchmark = _StubFlickr30kEntitiesBenchmark(dataset=dataset)

        report = benchmark.run(
            model=_DetectionModel("yellow helmet: [0.25, 0.125, 0.2, 0.25]"),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["mean_iou_matched"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
