from __future__ import annotations

import ast
import inspect
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from benchmarks.captioning import CaptioningBenchmark
from benchmarks.classification import ClassificationBenchmark
from benchmarks.detection import DetectionBenchmark
from benchmarks.multiple_choice import MultipleChoiceBenchmark


class _TrainingStubDataset:
    def __init__(self, rows: list[dict], labels: list[str] | None = None) -> None:
        self.name = "training_stub"
        self.rows = rows
        self.labels = list(labels or [])

    def get_samples(self, n: int) -> list[dict]:
        return self.rows[:n]

    def get_labels(self, rows) -> list[str]:
        del rows
        return self.labels

    def get_labels_img(self, row: dict) -> list[str]:
        return [str(row["label"])] if "label" in row else []

    def get_image_from_row(self, row: dict) -> Image.Image:
        return row["image"]

    def get_captions_from_row(self, row: dict) -> list[str]:
        return list(row.get("captions", []))

    def get_question_from_row(self, row: dict) -> str:
        return str(row.get("question", "Choose one."))

    def get_answer_from_row(self, row: dict) -> str:
        return str(row.get("answer", ""))

    def normalize_text(self, text: str) -> str:
        return str(text).strip().lower()


def _load_script(name: str) -> dict:
    directory = Path(__file__).resolve().parents[2] / "fine_tuning"
    sys.path.insert(0, str(directory))
    try:
        return runpy.run_path(str(directory / name))
    finally:
        sys.path.remove(str(directory))


class FineTuningPreparationTests(unittest.TestCase):
    def test_llava_image_grid_can_be_limited_for_training_memory(self) -> None:
        module = _load_script("train_llava_onevision_7b.py")

        class ImageProcessor:
            size = {"height": 384, "width": 384}
            image_grid_pinpoints = [
                [384, 384],
                [384, 768],
                [768, 768],
                [1152, 768],
            ]

        limited = module["limit_image_grid_pinpoints"](ImageProcessor(), 4)

        self.assertEqual(limited, [[384, 384], [384, 768], [768, 768]])

    def test_training_scripts_only_pass_supported_training_arguments(self) -> None:
        from transformers import TrainingArguments

        supported = set(inspect.signature(TrainingArguments).parameters)
        directory = Path(__file__).resolve().parents[2] / "fine_tuning"
        for script_name in ("train_llava_onevision_7b.py", "train_gemma4_31b.py"):
            tree = ast.parse((directory / script_name).read_text(encoding="utf-8"))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TrainingArguments"
            ]
            self.assertEqual(len(calls), 1, script_name)
            keywords = {
                keyword.arg for keyword in calls[0].keywords if keyword.arg is not None
            }
            self.assertFalse(
                keywords - supported, f"{script_name}: {keywords - supported}"
            )

    def test_fashion_mnist_balanced_indices_select_equal_classes_without_overlap(self) -> None:
        module = _load_script("prepare_fashion_mnist.py")
        select_balanced_indices = module["select_balanced_indices"]
        rows = [{"label": label} for label in range(3) for _ in range(6)]

        train_indices, validation_indices = select_balanced_indices(
            rows,
            label_count=3,
            train_per_class=2,
            validation_per_class=1,
            seed=42,
        )

        self.assertEqual(len(train_indices), 6)
        self.assertEqual(len(validation_indices), 3)
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(
            {label: sum(rows[index]["label"] == label for index in train_indices) for label in range(3)},
            {0: 2, 1: 2, 2: 2},
        )
        self.assertEqual(
            {label: sum(rows[index]["label"] == label for index in validation_indices) for label in range(3)},
            {0: 1, 1: 1, 2: 1},
        )

    def test_generic_preparer_exports_classification_and_caption_answers(self) -> None:
        module = _load_script("prepare_benchmark.py")
        export_examples = module["export_examples"]
        image = Image.new("RGB", (20, 20), "white")

        classification = ClassificationBenchmark(
            dataset=_TrainingStubDataset([{"image": image, "label": "coat"}], ["coat"]),
            name="classification",
        )
        classification_record = export_examples(
            benchmark=classification, count=1, label_sample_size=1
        )[0]
        self.assertEqual(classification_record["answer"], "coat")
        self.assertNotIn("USER:", classification_record["prompt"])

        captioning = CaptioningBenchmark(
            dataset=_TrainingStubDataset(
                [{"image": image, "captions": ["A white square.", "An empty tile."]}]
            ),
            name="captioning",
        )
        caption_record = export_examples(
            benchmark=captioning, count=1, label_sample_size=1
        )[0]
        self.assertEqual(caption_record["answer"], "A white square.")

    def test_generic_preparer_exports_multiple_choice_answer_and_detection_boxes(self) -> None:
        module = _load_script("prepare_benchmark.py")
        export_examples = module["export_examples"]
        image = Image.new("RGB", (100, 50), "white")
        multiple_choice = MultipleChoiceBenchmark(
            dataset=_TrainingStubDataset(
                [
                    {
                        "image": image,
                        "question": "Which description matches?",
                        "answer": "white square",
                        "choices": ["white square", "black square"],
                    }
                ]
            ),
            name="multiple_choice",
        )
        choice_record = export_examples(
            benchmark=multiple_choice, count=1, label_sample_size=1
        )[0]
        self.assertEqual(choice_record["answer"], "white square")
        self.assertIn("A. white square", choice_record["prompt"])

        row = {
            "image": image,
            "annotations": [{"label": "car", "bbox": [10.0, 5.0, 30.0, 10.0]}],
        }
        detection_dataset = _TrainingStubDataset([row], ["car"])
        detection_dataset.get_labels_img = lambda record: ["car"]
        detection = DetectionBenchmark(dataset=detection_dataset, name="detection")
        detection_record = export_examples(
            benchmark=detection, count=1, label_sample_size=1
        )[0]
        self.assertEqual(
            detection_record["answer"],
            "car: [0.100000, 0.100000, 0.300000, 0.200000]",
        )

    def test_generic_preparer_same_split_uses_disjoint_row_slices(self) -> None:
        module = _load_script("prepare_benchmark.py")
        build_records = module["build_records"]

        class StubBenchmark(ClassificationBenchmark):
            def __init__(self, split: str, streaming: bool) -> None:
                del split
                del streaming
                rows = [
                    {"image": Image.new("RGB", (4, 4), "white"), "label": label}
                    for label in ("a", "b", "c")
                ]
                super().__init__(
                    dataset=_TrainingStubDataset(rows, ["a", "b", "c"]),
                    name="stub",
                )

        train, validation = build_records(
            benchmark_cls=StubBenchmark,
            train_split="train",
            validation_split=None,
            train_examples=2,
            validation_examples=1,
            label_sample_size=3,
            streaming=True,
        )
        train_answers = [record["answer"] for record in train]
        validation_answers = [record["answer"] for record in validation]
        self.assertEqual(len(train_answers), 2)
        self.assertEqual(len(validation_answers), 1)
        self.assertFalse(set(train_answers) & set(validation_answers))
        self.assertEqual(set(train_answers + validation_answers), {"a", "b", "c"})

    def test_generic_preparer_can_reserve_an_evaluation_prefix(self) -> None:
        module = _load_script("prepare_benchmark.py")
        export_examples = module["export_examples"]
        image = Image.new("RGB", (4, 4), "white")
        benchmark = CaptioningBenchmark(
            dataset=_TrainingStubDataset(
                [
                    {"image": image, "captions": [caption]}
                    for caption in ("reserved", "train-a", "train-b")
                ]
            ),
            name="captioning",
        )

        records = export_examples(
            benchmark=benchmark,
            count=2,
            label_sample_size=3,
            skip_examples=1,
        )

        self.assertEqual(
            [record["answer"] for record in records],
            ["train-a", "train-b"],
        )

    def test_generic_preparer_writes_manifest_with_system_prompt(self) -> None:
        module = _load_script("prepare_benchmark.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = module["write_manifest"](
                records=[
                    {
                        "image": Image.new("RGB", (4, 4), "white"),
                        "prompt": "Return a label.",
                        "answer": "label",
                    }
                ],
                prefix="train",
                output_dir=Path(temp_dir),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["answer"], "label")
        self.assertEqual(payload["system"], module["SYSTEM_PROMPT"])

    def test_generic_preparer_registry_exposes_other_task_families(self) -> None:
        module = _load_script("prepare_benchmark.py")
        benchmarks = module["BENCHMARK_CLASSES"]
        for name in ("flickr30k", "docvqa", "openimages_v4_detection", "ucf101"):
            self.assertIn(name, benchmarks)


if __name__ == "__main__":
    unittest.main()
