from __future__ import annotations

import unittest
from typing import Any, Dict, List

from PIL import Image

from benchmarks import (
    INaturalistDetectionBenchmark,
    OpenVid1MCaptionBenchmark,
    ShareGPT4oImageEditBenchmark,
    VisualCoTDetectionBenchmark,
)
from dataset.visual_cot import VisualCoT


def _image(color: str = "white") -> Image.Image:
    return Image.new("RGB", (100, 80), color)


class _Dataset:
    def __init__(self, name: str, rows: List[Dict[str, Any]]) -> None:
        self.name = name
        self.rows = rows
        self.labels = []

    def get_samples(self, n: int):
        return self.rows[:n]

    def get_labels(self, rows):
        labels = []
        for row in rows:
            labels.extend(self.get_labels_img(row))
        return list(dict.fromkeys(labels))

    def get_labels_img(self, row):
        label = row.get("species_name", row.get("label", ""))
        return [label] if label else []

    def get_annotations_for_row(self, row):
        return list(row.get("annotations", []))

    def get_image_from_row(self, row):
        return row.get("image", row.get("source_image"))

    def get_captions_from_row(self, row):
        return list(row.get("captions", []))

    def get_question_from_row(self, row):
        return str(row.get("question", ""))

    def get_answer_from_row(self, row):
        return str(row.get("answer", ""))

    def get_choices_from_row(self, row):
        return list(row.get("choices", []))

    @staticmethod
    def normalize_text(text):
        return " ".join(str(text).strip().lower().split())


class _Model:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, image, prompt):
        del image
        del prompt
        return self.prediction


class MissingDatasetTypeTests(unittest.TestCase):
    def test_inaturalist_detection_scores_every_instance_of_one_species(self) -> None:
        row = {
            "id": 1,
            "image": _image(),
            "species_name": "Platalea ajaja",
            "annotations": [
                {"label": "Platalea ajaja", "bbox": [0.1, 0.2, 0.2, 0.3]},
                {"label": "Platalea ajaja", "bbox": [0.6, 0.1, 0.2, 0.2]},
            ],
        }
        benchmark = INaturalistDetectionBenchmark(
            dataset=_Dataset("inaturalist_detection", [row])
        )
        report = benchmark.run(
            _Model("[0.1, 0.2, 0.2, 0.3]\n[0.6, 0.1, 0.2, 0.2]"),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertEqual(result["valid_labels"], ["Platalea ajaja"])
        self.assertEqual(len(result["ground_truth_boxes"]), 2)

    def test_visual_cot_detection_parses_the_grounding_box(self) -> None:
        dataset = VisualCoT.__new__(VisualCoT)
        dataset.name = "visual_cot"
        dataset.labels = []
        row = {
            "image": _image(),
            "question": "What are the people looking at?",
            "conversations": [
                {"from": "gpt", "value": "[0.2, 0.3, 0.6, 0.8]"},
                {"from": "gpt", "value": "They are looking at their hands."},
            ],
        }
        benchmark = VisualCoTDetectionBenchmark(dataset=dataset)
        annotations = dataset.get_annotations_for_row(row)
        prompt = benchmark.make_prompt(["answer-relevant region"], row=row)

        self.assertEqual(annotations[0]["label"], "answer-relevant region")
        for actual, expected in zip(annotations[0]["bbox"], [0.2, 0.3, 0.4, 0.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertIn(row["question"], prompt)
        correct, labels, metrics = benchmark.evaluate_prediction(
            row,
            "[0.2, 0.3, 0.4, 0.5]",
            image=row["image"],
        )
        self.assertTrue(correct)
        self.assertEqual(labels, ["answer-relevant region"])
        self.assertEqual(metrics["f1"], 1.0)

    def test_openvid_caption_uses_the_video_description_as_reference(self) -> None:
        row = {
            "image": _image(),
            "captions": ["A person walks beside the ocean at sunset."],
        }
        benchmark = OpenVid1MCaptionBenchmark(dataset=_Dataset("openvid1m", [row]))
        report = benchmark.run(
            _Model(row["captions"][0]),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        self.assertTrue(report["results"][0]["correct"])
        self.assertEqual(report["results"][0]["reference_captions"], row["captions"])

    def test_sharegpt_edit_uses_before_and_after_images(self) -> None:
        answer = "replace the red square with a green square"
        row = {
            "source_image": _image("red"),
            "target_image": _image("green"),
            "answer": answer,
        }
        benchmark = ShareGPT4oImageEditBenchmark(
            dataset=_Dataset("sharegpt4o_image_edit", [row])
        )
        prepared, _ = benchmark.prepare(n=1, label_sample_size=1)
        prompt_labels = benchmark.get_prompt_labels_for_row(prepared[0], [])
        prompt = benchmark.make_prompt(prompt_labels, row=prepared[0])

        self.assertEqual(len(prompt_labels), 4)
        self.assertIn("Image A (before)", prompt)
        self.assertEqual(benchmark.get_image_for_row(prepared[0]).size, (200, 104))
        report = benchmark.run(
            _Model(answer),
            n=1,
            label_sample_size=1,
            show_progress=False,
        )
        self.assertTrue(report["results"][0]["correct"])


if __name__ == "__main__":
    unittest.main()
