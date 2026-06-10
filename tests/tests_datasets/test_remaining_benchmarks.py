from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List
from unittest.mock import patch

from PIL import Image

from benchmarks import (
    BLIP3o60kBenchmark,
    Flickr30kEntitiesBenchmark,
    LVISBenchmark,
    OpenImagesV4DetectionBenchmark,
    VisualCoTBenchmark,
    VisualGenomeBenchmark,
)


class _BaseStubDataset:
    def __init__(self, name: str, rows: List[Dict[str, Any]], labels: List[str] | None = None) -> None:
        self.name = name
        self._rows = rows
        self.labels = list(labels or [])

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self._rows[:n]

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("image") is not None:
            return row["image"]
        if row.get("frames"):
            return row["frames"][0]
        if row.get("source_image") is not None:
            return row["source_image"]
        raise KeyError("row does not contain image data")

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        label = row.get("label_text", row.get("label"))
        return [str(label)] if label is not None else []

    def normalize_text(self, text: str) -> str:
        text = str(text).strip().lower()
        for ch in [".", ",", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}"]:
            text = text.replace(ch, "")
        return " ".join(text.split())


class _QADataset(_BaseStubDataset):
    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        return str(row["question"])

    def get_answers_from_row(self, row: Dict[str, Any]) -> List[str]:
        return list(row["answers"])


class _ChoiceDataset(_BaseStubDataset):
    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        return str(row["question"])

    def get_choices_from_row(self, row: Dict[str, Any]) -> List[str]:
        return list(row["choices"])

    def get_answer_from_row(self, row: Dict[str, Any]) -> str:
        return str(row["answer"])


class _PromptAwareModel:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, image: Image.Image, prompt: str) -> str:
        del image
        del prompt
        return self.prediction


def _square(color: str) -> Image.Image:
    return Image.new("RGB", (24, 24), color)


def _frames() -> List[Image.Image]:
    return [_square("red"), _square("green"), _square("blue")]


class RemainingBenchmarkTests(unittest.TestCase):
    def test_remaining_benchmarks_run_on_stub_datasets(self) -> None:
        cases = [
            (VisualGenomeBenchmark, _QADataset("visual_genome", [{"image": _square("white"), "question": "What relationship is shown?", "answers": ["person riding horse"]}]), _PromptAwareModel("person riding horse")),
            (LVISBenchmark, _BaseStubDataset("lvis", [{"image": _square("white"), "objects": [{"label": "rare object", "bbox": [1, 1, 8, 8]}]}], labels=["rare object"]), _PromptAwareModel("rare object: [0.04, 0.04, 0.33, 0.33]")),
            (VisualCoTBenchmark, _QADataset("visual_cot", [{"image": _square("orange"), "question": "How many shapes are visible?", "answers": ["1"]}]), _PromptAwareModel("1")),
            (BLIP3o60kBenchmark, _ChoiceDataset("blip3o_60k", [{"image": _square("green"), "question": "Which prompt generated the image?", "choices": ["a green square", "a tree", "a red circle", "a blue sky"], "answer": "a green square"}]), _PromptAwareModel("a green square")),
            (Flickr30kEntitiesBenchmark, _BaseStubDataset("flickr30k_entities", [{"image": _square("white"), "annotations": [{"label": "helmet", "bbox": [1, 1, 8, 8]}]}], labels=["helmet"]), _PromptAwareModel("helmet: [0.04, 0.04, 0.29, 0.29]")),
            (OpenImagesV4DetectionBenchmark, _BaseStubDataset("openimages_v4_detection", [{"image": _square("white"), "annotations": [{"label": "car", "bbox": [1, 1, 8, 8]}]}], labels=["car"]), _PromptAwareModel("car: [0.04, 0.04, 0.29, 0.29]")),
        ]

        for benchmark_cls, dataset, model in cases:
            with self.subTest(benchmark=benchmark_cls.__name__):
                benchmark = benchmark_cls(dataset=dataset)
                report = benchmark.run(model=model, n=1, label_sample_size=2, show_progress=False)
                self.assertEqual(report["num_samples"], 1)
                self.assertTrue(report["results"][0]["correct"])

    def test_remaining_benchmarks_have_default_loader(self) -> None:
        loader_cases = [
            ("benchmarks.visual_qa.visual_genome.VisualGenome", VisualGenomeBenchmark),
            ("benchmarks.detection.lvis.LVIS", LVISBenchmark),
            ("benchmarks.visual_qa.visual_cot.VisualCoT", VisualCoTBenchmark),
            ("benchmarks.multiple_choice.blip3o_60k.BLIP3o60k", BLIP3o60kBenchmark),
            ("benchmarks.detection.flickr30k_entities.Flickr30kEntities", Flickr30kEntitiesBenchmark),
            ("benchmarks.detection.openimages_v4_detection.OpenImagesV4", OpenImagesV4DetectionBenchmark),
        ]

        for target, benchmark_cls in loader_cases:
            del target
            fake_dataset = _BaseStubDataset("fake", [{"image": _square("white"), "label": "label"}], labels=["label"])
            with self.subTest(benchmark=benchmark_cls.__name__):
                with patch.object(benchmark_cls, "dataset_cls", return_value=fake_dataset) as loader_cls:
                    benchmark = benchmark_cls()
                    self.assertIs(benchmark.dataset, fake_dataset)
                    loader_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
