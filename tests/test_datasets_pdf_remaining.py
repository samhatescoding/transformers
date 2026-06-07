from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List
from unittest.mock import patch

from PIL import Image

from benchmarks import (
    CityscapesBenchmark,
    DiffusionDBBenchmark,
    FlyingThings3DBenchmark,
    HDTFBenchmark,
    HQEditBenchmark,
    ImgEditBenchmark,
    MagicBrushBenchmark,
    PickAPicBenchmark,
    ShareGPT4oImageBenchmark,
    TAD66KBenchmark,
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


class DatasetsPdfRemainingTests(unittest.TestCase):
    def test_remaining_pdf_benchmarks_run_on_stub_datasets(self) -> None:
        pair_row = {
            "source_image": _square("red"),
            "target_image": _square("green"),
            "question": "Which edit happened?",
            "choices": ["change the object to green", "add a tree"],
            "answer": "change the object to green",
        }
        cases = [
            (FlyingThings3DBenchmark, _ChoiceDataset("flyingthings3d", [{"image": _square("blue"), "question": "Which scene matches?", "choices": ["floating 3D objects", "a face video"], "answer": "floating 3D objects"}]), _PromptAwareModel("A")),
            (CityscapesBenchmark, _BaseStubDataset("cityscapes", [{"image": _square("gray"), "label": "road"}], labels=["road", "sky"]), _PromptAwareModel("road")),
            (HDTFBenchmark, _ChoiceDataset("hdtf", [{"frames": _frames(), "question": "Which transcript matches?", "choices": ["hello there", "quiet forest"], "answer": "hello there"}]), _PromptAwareModel("A")),
            (DiffusionDBBenchmark, _ChoiceDataset("diffusiondb", [{"image": _square("purple"), "question": "Which prompt matches?", "choices": ["purple square", "green field"], "answer": "purple square"}]), _PromptAwareModel("A")),
            (TAD66KBenchmark, _BaseStubDataset("tad66k", [{"image": _square("yellow"), "label": "landscape"}], labels=["landscape", "portrait"]), _PromptAwareModel("landscape")),
            (MagicBrushBenchmark, _ChoiceDataset("magicbrush", [dict(pair_row)]), _PromptAwareModel("A")),
            (PickAPicBenchmark, _ChoiceDataset("pick_a_pic", [{"image": _square("orange"), "question": "Which prompt matches?", "choices": ["orange square", "blue square"], "answer": "orange square"}]), _PromptAwareModel("A")),
            (HQEditBenchmark, _ChoiceDataset("hq_edit", [dict(pair_row)]), _PromptAwareModel("A")),
            (ImgEditBenchmark, _ChoiceDataset("imgedit", [dict(pair_row)]), _PromptAwareModel("A")),
            (ShareGPT4oImageBenchmark, _ChoiceDataset("sharegpt4o_image", [dict(pair_row)]), _PromptAwareModel("A")),
        ]

        for benchmark_cls, dataset, model in cases:
            with self.subTest(benchmark=benchmark_cls.__name__):
                benchmark = benchmark_cls(dataset=dataset)
                report = benchmark.run(model=model, n=1, label_sample_size=2, show_progress=False)
                self.assertEqual(report["num_samples"], 1)
                self.assertTrue(report["results"][0]["correct"])

    def test_remaining_pdf_benchmarks_have_default_loader(self) -> None:
        benchmark_classes = [
            FlyingThings3DBenchmark,
            CityscapesBenchmark,
            HDTFBenchmark,
            DiffusionDBBenchmark,
            TAD66KBenchmark,
            MagicBrushBenchmark,
            PickAPicBenchmark,
            HQEditBenchmark,
            ImgEditBenchmark,
            ShareGPT4oImageBenchmark,
        ]

        for benchmark_cls in benchmark_classes:
            fake_dataset = _BaseStubDataset("fake", [{"image": _square("white"), "label": "label"}], labels=["label"])
            with self.subTest(benchmark=benchmark_cls.__name__):
                with patch.object(benchmark_cls, "dataset_cls", return_value=fake_dataset) as loader_cls:
                    benchmark = benchmark_cls()
                    self.assertIs(benchmark.dataset, fake_dataset)
                    loader_cls.assert_called_once_with(split=benchmark_cls.default_split, streaming=True)


if __name__ == "__main__":
    unittest.main()
