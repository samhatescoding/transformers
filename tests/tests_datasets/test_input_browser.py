from __future__ import annotations

import unittest
import runpy
import sys
import tempfile
from unittest.mock import Mock, patch

from PIL import Image

from benchmarks.curated_answer_choices import _load_dataset
from benchmarks.captioning import CaptioningBenchmark
from benchmarks.detection import DetectionBenchmark
from benchmarks.visual_qa._visual_qa import VisualQABenchmark
from dataset._base_dataset import BaseDataset
from dataset.mscoco import MSCOCO
from ui.input_browser import (
    BENCHMARK_SPECS,
    BENCHMARK_TYPES,
    BenchmarkInputBrowser,
    BenchmarkInputService,
    BenchmarkSpec,
    ModelInputPreview,
)
from ui.input_browser import ROOT_DIR


class FakeVQADataset(BaseDataset):
    name = "vqav2"
    dataset_id = "fake/vqav2"
    split = "validation"
    labels = []

    def __iter__(self):
        return iter(self.get_samples(1))

    def get_samples(self, n):
        rows = [
            {
                "image": Image.new("RGB", (32, 24), "white"),
                "question": "Where are the kids riding?",
                "answers": ["carnival ride"],
            }
        ]
        return rows[:n]

    def get_image_from_row(self, row):
        return row["image"]

    def get_question_from_row(self, row):
        return row["question"]

    def get_answers_from_row(self, row):
        return row["answers"]

    def get_labels(self, rows):
        return []

    def get_labels_img(self, row):
        return row["answers"]


class FakeVQABenchmark(VisualQABenchmark):
    benchmark_name = "fake_vqav2"


class FakeDetectionDataset(BaseDataset):
    name = "fake_detection"
    dataset_id = "fake/detection"
    split = "validation"
    labels = ["sink"]

    def __iter__(self):
        return iter(self.get_samples(1))

    def get_samples(self, n):
        return [
            {
                "id": "detection-row",
                "image": Image.new("RGB", (100, 80), "white"),
                "annotations": [
                    {"label": "sink", "bbox": [10.0, 20.0, 30.0, 40.0]},
                    {"label": "sink", "bbox": [60.0, 10.0, 20.0, 20.0]},
                ],
            }
        ][:n]

    def get_image_from_row(self, row):
        return row["image"]

    def get_labels(self, rows):
        del rows
        return list(self.labels)

    def get_labels_img(self, row):
        return [item["label"] for item in row["annotations"]]

    def get_annotations_for_row(self, row):
        return list(row["annotations"])


class FakeDetectionBenchmark(DetectionBenchmark):
    benchmark_name = "fake_detection"


class FakeCaptionDataset(BaseDataset):
    name = "fake_caption"
    dataset_id = "fake/caption"
    split = "validation"
    labels = []

    def __iter__(self):
        return iter(self.get_samples(1))

    def get_samples(self, n):
        return [
            {
                "image": Image.new("RGB", (40, 30), "white"),
                "captions": ["A white rectangle.", "A blank white image."],
            }
        ][:n]

    def get_labels(self, rows):
        del rows
        return []

    def get_labels_img(self, row):
        del row
        return []

    def get_image_from_row(self, row):
        return row["image"]

    def get_captions_from_row(self, row):
        return list(row["captions"])


class FakeCaptionBenchmark(CaptioningBenchmark):
    benchmark_name = "fake_caption"


class FakeWidget:
    def __init__(self):
        self.config = {}
        self.content = ""
        self.state = None

    def configure(self, **kwargs):
        self.config.update(kwargs)
        if "state" in kwargs:
            self.state = kwargs["state"]

    def delete(self, _start, _end):
        self.content = ""

    def insert(self, _index, value):
        self.content += str(value)

    def see(self, _index):
        return None

    def winfo_width(self):
        return 640

    def winfo_height(self):
        return 480

    def place_forget(self):
        self.config["placed"] = False

    def pack_forget(self):
        self.config["packed"] = False


class FakeVariable:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = str(value)


class InputBrowserTests(unittest.TestCase):
    def test_type_a_choice_files_have_twenty_valid_rows(self):
        for dataset_name in (
            "flyingthings3d",
            "visual_genome",
            "vqav2",
            "gqa",
            "docvqa",
            "visual_cot",
        ):
            with self.subTest(dataset=dataset_name):
                rows = _load_dataset(dataset_name)
                self.assertEqual(len(rows), 20)
                for row_index, row in enumerate(rows):
                    self.assertEqual(row["row_index"], row_index)
                    if row.get("annotation_status") == "draft":
                        self.assertEqual(dataset_name, "flyingthings3d")
                        self.assertEqual(row["choices"], ["above", "below", "right", "left"])
                        self.assertIsNone(row["answer"])
                        self.assertIsNone(row["correct_choice_index"])
                        continue
                    normalized_answer = str(row["answer"]).strip().casefold()
                    matches = [
                        index
                        for index, choice in enumerate(row["choices"])
                        if str(choice).strip().casefold() == normalized_answer
                    ]
                    self.assertEqual(matches, [row["correct_choice_index"]])

    def test_type_registry_contains_exactly_eight_types_and_valid_specs(self):
        codes = [item.code for item in BENCHMARK_TYPES]
        self.assertEqual(codes, ["A", "B", "C", "E", "G", "L", "P", "R"])
        self.assertEqual({spec.type_code for spec in BENCHMARK_SPECS}, set(codes))

    def test_type_l_has_one_complete_unique_label_file_per_benchmark(self):
        expected_counts = {
            "cityscapes.txt": 19,
            "fairface.txt": 9,
            "fashion_mnist.txt": 10,
            "imagenet1k.txt": 1000,
            "inaturalist2017.txt": 4895,
            "lsun.txt": 10,
            "mvtec_ad.txt": 2,
            "openimages_v4.txt": 601,
            "places365.txt": 365,
            "dfdc.txt": 2,
            "kinetics700.txt": 700,
            "ucf101.txt": 101,
        }
        labels_root = ROOT_DIR / "benchmark_choices" / "type_l"
        actual_files = {path.name for path in labels_root.glob("*.txt")}
        self.assertEqual(actual_files, set(expected_counts))

        for filename, expected_count in expected_counts.items():
            with self.subTest(filename=filename):
                labels = (labels_root / filename).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(labels), expected_count)
                self.assertTrue(all(label.strip() == label and label for label in labels))
                self.assertEqual(len({label.casefold() for label in labels}), expected_count)

    def test_visual_qa_prepare_uses_curated_choices_and_exact_prompt(self):
        benchmark = FakeVQABenchmark(dataset=FakeVQADataset())
        rows, labels = benchmark.prepare(n=1, label_sample_size=64)
        prompt_labels = benchmark.get_prompt_labels_for_row(rows[0], labels)
        prompt = benchmark.make_prompt(prompt_labels, row=rows[0], image=rows[0]["image"])

        self.assertEqual(prompt_labels, ["carnival ride", "playground", "school bus", "bicycle"])
        self.assertIn("A. carnival ride", prompt)
        self.assertIn("D. bicycle", prompt)
        self.assertTrue(benchmark.evaluate_prediction(rows[0], "A")[0])

    def test_visual_qa_skips_stale_curated_row_instead_of_using_wrong_choices(self):
        dataset = FakeVQADataset()
        dataset.name = "visual_cot"
        benchmark = FakeVQABenchmark(dataset=dataset)
        rows, labels = benchmark.prepare(n=1, label_sample_size=1)
        prompt_labels = benchmark.get_prompt_labels_for_row(rows[0], labels)
        prompt = benchmark.make_prompt(prompt_labels, row=rows[0], image=rows[0]["image"])

        self.assertEqual(prompt_labels, [])
        self.assertNotIn("Choices:", prompt)

    def test_input_service_calls_runtime_input_methods(self):
        benchmark = FakeVQABenchmark(dataset=FakeVQADataset())
        spec = BenchmarkSpec("A", "Fake", "fake.module", "FakeBenchmark")
        service = BenchmarkInputService(sample_count=1)

        class FakeModule:
            FakeBenchmark = staticmethod(lambda streaming=True: benchmark)

        events = []
        with patch("ui.input_browser.importlib.import_module", return_value=FakeModule):
            preview = service.preview(
                spec,
                0,
                progress=lambda step, total, message: events.append((step, total, message)),
            )

        self.assertEqual(preview.image.size, (32, 24))
        self.assertEqual(preview.source, "fake/vqav2")
        self.assertTrue(preview.prompt.startswith("USER: <image>"))
        self.assertIn("Choices:", preview.prompt)
        self.assertEqual([event[0] for event in events], list(range(1, 9)))
        self.assertTrue(all(event[1] == 8 for event in events))

    def test_cached_preview_reports_cached_preparation_steps(self):
        benchmark = FakeVQABenchmark(dataset=FakeVQADataset())
        spec = BenchmarkSpec("A", "Fake", "fake.module", "FakeBenchmark")
        service = BenchmarkInputService(sample_count=1)

        class FakeModule:
            FakeBenchmark = staticmethod(lambda streaming=True: benchmark)

        with patch("ui.input_browser.importlib.import_module", return_value=FakeModule):
            service.preview(spec, 0)
            events = []
            service.preview(
                spec,
                0,
                progress=lambda step, total, message: events.append((step, total, message)),
            )

        self.assertIn("in-memory cache", events[2][2])
        self.assertEqual([event[0] for event in events], list(range(1, 9)))

    def test_type_b_preview_draws_selected_ground_truth_boxes(self):
        benchmark = FakeDetectionBenchmark(dataset=FakeDetectionDataset())
        spec = BenchmarkSpec("B", "Fake Detection", "fake.module", "FakeDetectionBenchmark")
        service = BenchmarkInputService(sample_count=1)

        class FakeModule:
            FakeDetectionBenchmark = staticmethod(lambda streaming=True: benchmark)

        with patch("ui.input_browser.importlib.import_module", return_value=FakeModule):
            preview = service.preview(spec, 0)

        self.assertEqual(preview.correct_answers, ["sink"])
        self.assertFalse(preview.show_correct_answer)
        self.assertEqual(preview.displayed_box_count, 2)
        self.assertEqual(preview.image.getpixel((10, 20)), (255, 59, 48))
        self.assertEqual(preview.image.getpixel((60, 10)), (255, 59, 48))

    def test_type_c_preview_shows_reference_captions(self):
        benchmark = FakeCaptionBenchmark(dataset=FakeCaptionDataset())
        spec = BenchmarkSpec("C", "Fake Caption", "fake.module", "FakeCaptionBenchmark")
        service = BenchmarkInputService(sample_count=1)

        class FakeModule:
            FakeCaptionBenchmark = staticmethod(lambda streaming=True: benchmark)

        with patch("ui.input_browser.importlib.import_module", return_value=FakeModule):
            preview = service.preview(spec, 0)

        self.assertTrue(preview.show_correct_answer)
        self.assertEqual(
            preview.correct_answers,
            ["A white rectangle.", "A blank white image."],
        )

    def test_direct_script_bootstraps_repository_import_path(self):
        script_path = ROOT_DIR / "ui" / "input_browser.py"
        original_path = list(sys.path)
        try:
            sys.path[:] = [
                entry
                for entry in sys.path
                if entry and entry != str(ROOT_DIR)
            ]
            with tempfile.TemporaryDirectory() as temporary_directory:
                with patch("os.getcwd", return_value=temporary_directory):
                    namespace = runpy.run_path(str(script_path), run_name="input_browser_test")
            self.assertEqual(namespace["ROOT_DIR"], ROOT_DIR)
            self.assertEqual(sys.path[0], str(ROOT_DIR))
        finally:
            sys.path[:] = original_path

    def test_mscoco_annotation_cache_is_repository_relative(self):
        self.assertEqual(
            MSCOCO.ANNOTATION_CACHE_DIR,
            ROOT_DIR / ".tmp" / "coco_annotations",
        )

    def test_every_benchmark_square_wires_and_renders_a_synthetic_model_input(self):
        browser = BenchmarkInputBrowser.__new__(BenchmarkInputBrowser)
        browser._request_id = 1
        browser._show_preview = Mock()

        for spec in BENCHMARK_SPECS:
            with self.subTest(square=spec.name, type=spec.type_code):
                browser._benchmark_click_command(spec)()
                browser._show_preview.assert_called_once_with(spec, 0)
                browser._show_preview.reset_mock()

        browser.image_label = FakeWidget()
        browser.prompt_text = FakeWidget()
        browser.status_var = FakeVariable()
        browser.status_label = FakeWidget()
        browser.meta_var = FakeVariable()
        browser.answer_text = FakeWidget()
        browser.answer_panel = FakeWidget()
        browser.progress_bar = FakeWidget()
        browser.loading_log = FakeWidget()
        browser.loading_panel = FakeWidget()
        browser.prev_button = FakeWidget()
        browser.next_button = FakeWidget()
        browser._photo = None
        browser._preview_image = None

        with patch("ui.input_browser.ImageTk.PhotoImage", side_effect=lambda image: image):
            for spec in BENCHMARK_SPECS:
                preview = ModelInputPreview(
                    benchmark_name=spec.name,
                    dataset_name=f"dataset_{spec.class_name}",
                    row_index=0,
                    row_count=1,
                    image=Image.new("RGB", (48, 32), "navy"),
                    prompt=f"USER: <image>\nExact prompt for {spec.name}\nASSISTANT:",
                    prompt_labels=["label"],
                    source=f"fake/{spec.class_name}",
                    split="test",
                    correct_answers=["correct label"],
                    show_correct_answer=spec.type_code != "B",
                )
                with self.subTest(render=spec.name, type=spec.type_code):
                    browser._display_preview(1, preview)
                    self.assertIsNotNone(browser.image_label.config.get("image"))
                    self.assertEqual(browser.image_label.config.get("text"), "")
                    self.assertEqual(browser.prompt_text.content, preview.prompt)
                    if spec.type_code == "B":
                        self.assertFalse(browser.answer_panel.config.get("packed", True))
                    else:
                        self.assertEqual(browser.answer_text.content, "correct label")
                    self.assertIn(spec.name, browser.meta_var.value)

    def test_long_answers_are_written_to_the_scrollable_answer_widget(self):
        browser = BenchmarkInputBrowser.__new__(BenchmarkInputBrowser)
        browser._request_id = 1
        browser.image_label = FakeWidget()
        browser.prompt_text = FakeWidget()
        browser.status_var = FakeVariable()
        browser.status_label = FakeWidget()
        browser.meta_var = FakeVariable()
        browser.answer_text = FakeWidget()
        browser.answer_panel = FakeWidget()
        browser.progress_bar = FakeWidget()
        browser.loading_log = FakeWidget()
        browser.loading_panel = FakeWidget()
        browser.prev_button = FakeWidget()
        browser.next_button = FakeWidget()
        browser._photo = None
        browser._preview_image = None
        captions = [f"Caption {index}: " + ("long text " * 20) for index in range(20)]
        preview = ModelInputPreview(
            benchmark_name="InternVid",
            dataset_name="internvid",
            row_index=0,
            row_count=1,
            image=Image.new("RGB", (48, 32), "navy"),
            prompt="USER: <image>\nWrite a caption.\nASSISTANT:",
            prompt_labels=[],
            source="OpenGVLab/InternVid-Full",
            split="train",
            correct_answers=captions,
        )

        with patch("ui.input_browser.ImageTk.PhotoImage", side_effect=lambda image: image):
            browser._display_preview(1, preview)

        self.assertEqual(browser.prompt_text.content, preview.prompt)
        self.assertEqual(browser.answer_text.content, "\n\n".join(captions))

    def test_preview_fit_preserves_aspect_ratio_without_cropping(self):
        wide = Image.new("RGB", (1920, 540), "navy")
        fitted = BenchmarkInputBrowser._fit_image(wide, 640, 480)
        self.assertEqual(fitted.size, (640, 180))

        tall = Image.new("RGB", (400, 1200), "navy")
        fitted = BenchmarkInputBrowser._fit_image(tall, 640, 480)
        self.assertEqual(fitted.size, (160, 480))


if __name__ == "__main__":
    unittest.main()
