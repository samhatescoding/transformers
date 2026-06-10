from __future__ import annotations

import unittest
from typing import Any, Dict, List

from PIL import Image

from benchmarks import (
    AestheticRatingBenchmark,
    HQEditBenchmark,
    PickAPicBenchmark,
    PromptReconstructionBenchmark,
)
from benchmarks.curated_distractors import get_curated_distractors
from dataset.pick_a_pic import PickAPic
from dataset.tad66k import TAD66K


def _image(color: str) -> Image.Image:
    return Image.new("RGB", (20, 16), color)


class _Dataset:
    def __init__(self, name: str, rows: List[Dict[str, Any]]) -> None:
        self.name = name
        self.rows = rows

    def get_samples(self, n: int):
        return self.rows[:n]

    def get_labels(self, rows):
        del rows
        return []

    def get_labels_img(self, row):
        answer = row.get("answer", row.get("rating"))
        return [str(answer)] if answer is not None else []

    def get_image_from_row(self, row):
        return row.get("image", row.get("source_image"))

    def get_question_from_row(self, row):
        return row.get("question", "")

    def get_answer_from_row(self, row):
        return row.get("answer", "")

    def get_choices_from_row(self, row):
        return row.get("choices", [])

    def get_rating_from_row(self, row):
        return row.get("rating")

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(str(text).strip().lower().split())


class _Model:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction

    def predict(self, image, prompt):
        del image
        del prompt
        return self.prediction


class NewDatasetTaskTypeTests(unittest.TestCase):
    def test_image_modification_uses_pair_and_four_shuffled_instructions(self) -> None:
        answer = "turn the red square green"
        row = {
            "id": "edit-1",
            "source_image": _image("red"),
            "target_image": _image("green"),
            "answer": answer,
            "choices": [answer, "add a tree", "remove the square", "make it monochrome"],
        }
        benchmark = HQEditBenchmark(dataset=_Dataset("hq_edit", [row]))
        rows, _ = benchmark.prepare(n=1, label_sample_size=1)

        self.assertEqual(len(rows[0]["choices"]), 4)
        self.assertCountEqual(
            rows[0]["choices"],
            [answer, *get_curated_distractors("hq_edit", 0, answer)],
        )
        self.assertEqual(benchmark.get_image_for_row(rows[0]).size, (40, 40))

        report = benchmark.run(_Model(answer), n=1, label_sample_size=1, show_progress=False)
        self.assertTrue(report["results"][0]["correct"])

    def test_prompt_reconstruction_requires_four_prompts(self) -> None:
        answer = "a purple castle at sunset"
        row = {
            "id": "generated-1",
            "image": _image("purple"),
            "answer": answer,
            "choices": [answer, "a city bus", "a handwritten invoice", "a bowl of fruit"],
        }
        benchmark = PromptReconstructionBenchmark(dataset=_Dataset("diffusiondb", [row]), name="diffusiondb")
        rows, _ = benchmark.prepare(n=1, label_sample_size=1)

        self.assertEqual(len(rows[0]["choices"]), 4)
        self.assertCountEqual(
            rows[0]["choices"],
            [answer, *get_curated_distractors("diffusiondb", 0, answer)],
        )
        report = benchmark.run(_Model(answer), n=1, label_sample_size=1, show_progress=False)
        self.assertTrue(report["results"][0]["correct"])

    def test_prompt_datasets_have_twenty_curated_distractor_rows(self) -> None:
        for dataset_name in (
            "hq_edit",
            "imgedit",
            "magicbrush",
            "sharegpt4o_image_edit",
            "diffusiondb",
            "blip3o_60k",
            "conceptual_captions",
            "openvid1m",
            "sharegpt4o_image",
        ):
            with self.subTest(dataset=dataset_name):
                for row_index in range(20):
                    distractors = get_curated_distractors(
                        dataset_name,
                        row_index,
                        "a red car parked on a rainy city street",
                    )
                    self.assertEqual(len(distractors), 3)
                    self.assertEqual(len(set(distractors)), 3)
                self.assertEqual(
                    get_curated_distractors(
                        dataset_name,
                        20,
                        "a red car parked on a rainy city street",
                    ),
                    (),
                )

    def test_curated_distractors_form_balanced_two_attribute_combinations(self) -> None:
        answer = "a red car parked on a rainy city street"
        choices = [
            answer,
            *get_curated_distractors("diffusiondb", 0, answer),
        ]

        self.assertCountEqual(
            choices,
            [
                "a red car parked on a rainy city street",
                "a blue car parked on a rainy city street",
                "a red car parked on a sunny city street",
                "a blue car parked on a sunny city street",
            ],
        )
        self.assertEqual(sum("red car" in choice for choice in choices), 2)
        self.assertEqual(sum("blue car" in choice for choice in choices), 2)
        self.assertEqual(sum("rainy city" in choice for choice in choices), 2)
        self.assertEqual(sum("sunny city" in choice for choice in choices), 2)

    def test_image_preference_scores_image_a_or_b(self) -> None:
        row = {
            "image_a": _image("white"),
            "image_b": _image("black"),
            "answer": "Image B",
            "choices": ["Image A", "Image B"],
        }
        benchmark = PickAPicBenchmark(dataset=_Dataset("pick_a_pic", [row]))
        image = benchmark.get_image_for_row(row)

        self.assertEqual(image.size, (40, 40))
        report = benchmark.run(_Model("B"), n=1, label_sample_size=1, show_progress=False)
        self.assertTrue(report["results"][0]["correct"])

    def test_aesthetic_rating_reports_exact_accuracy_and_mae(self) -> None:
        row = {"image": _image("blue"), "rating": 7}
        benchmark = AestheticRatingBenchmark(dataset=_Dataset("tad66k", [row]), name="tad66k")

        report = benchmark.run(_Model("I would rate it 6."), n=1, label_sample_size=1, show_progress=False)

        self.assertFalse(report["results"][0]["correct"])
        self.assertEqual(report["results"][0]["absolute_error"], 1)
        self.assertEqual(report["stats"]["mean_absolute_error"], 1.0)

    def test_pick_a_pic_maps_human_preference_to_image_label(self) -> None:
        dataset = PickAPic.__new__(PickAPic)
        dataset.preference_keys = ("human_pref",)

        self.assertEqual(dataset.get_answer_from_row({"human_pref": 0}), "Image A")
        self.assertEqual(dataset.get_answer_from_row({"human_pref": 1}), "Image B")

    def test_tad66k_rounds_scores_to_nearest_rating(self) -> None:
        dataset = TAD66K.__new__(TAD66K)
        dataset.score_keys = ("score",)

        self.assertEqual(dataset.get_rating_from_row({"score": 6.6}), 7)
        self.assertEqual(dataset.get_rating_from_row({"score": 10.8}), 10)

    def test_tad66k_selects_rows_spaced_across_the_score_range(self) -> None:
        dataset = TAD66K.__new__(TAD66K)
        dataset.split = "train"
        dataset.score_keys = ("score",)
        dataset.rows = [{"score": float(score), "rating": score} for score in range(1, 11)]

        rows = dataset.get_score_spaced_samples(5)
        scores = [row["score"] for row in rows]

        self.assertEqual(sorted(scores), [1.0, 3.0, 5.0, 8.0, 10.0])
        self.assertNotEqual(scores, sorted(scores))
        self.assertEqual(scores, [3.0, 5.0, 8.0, 1.0, 10.0])


if __name__ == "__main__":
    unittest.main()
