from __future__ import annotations

import unittest
from typing import Any, Dict, Iterable, List
from unittest.mock import patch

from PIL import Image

from benchmarks import (
    ConceptualCaptionsBenchmark,
    DFDCBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    GQABenchmark,
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenVid1MBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    VQAv2Benchmark,
    MSCOCOCaptionBenchmark,
)
from benchmarks.captioning import CaptioningBenchmark
from dataset import FashionMNIST, HDTF, InternVid, LAION400M, LAION5B
from dataset.hf_common import HFCaptionDataset, HFVideoCaptionDataset


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


class _CaptionDataset(_BaseStubDataset):
    def get_captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        return list(row["captions"])


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


class AdditionalBenchmarkTests(unittest.TestCase):
    def test_all_new_benchmarks_run_on_stub_datasets(self) -> None:
        cases = [
            (LSUNBenchmark, _BaseStubDataset("lsun", [{"image": _square("white"), "label": "bedroom"}], labels=["bedroom", "classroom"]), _PromptAwareModel("bedroom")),
            (VQAv2Benchmark, _QADataset("vqav2", [{"image": _square("white"), "question": "What color is the square?", "answers": ["white"]}]), _PromptAwareModel("white")),
            (FashionMNISTBenchmark, _BaseStubDataset("fashion_mnist", [{"image": _square("gray"), "label": "coat"}], labels=["coat", "dress"]), _PromptAwareModel("coat")),
            (KineticsBenchmark, _BaseStubDataset("kinetics", [{"frames": _frames(), "label_text": "jumping"}], labels=["jumping", "running"]), _PromptAwareModel("jumping")),
            (PlacesBenchmark, _BaseStubDataset("places", [{"image": _square("yellow"), "label": "kitchen"}], labels=["kitchen", "forest"]), _PromptAwareModel("kitchen")),
            (ConceptualCaptionsBenchmark, _ChoiceDataset("conceptual_captions", [{"image": _square("orange"), "question": "Which caption matches the image?", "choices": ["A cat on a mat", "A dog in a field"], "answer": "A dog in a field"}]), _PromptAwareModel("A dog in a field")),
            (GQABenchmark, _QADataset("gqa", [{"image": _square("blue"), "question": "What color is the square?", "answers": ["blue"]}]), _PromptAwareModel("blue")),
            (MVTecADBenchmark, _BaseStubDataset("mvtec_ad", [{"image": _square("black"), "label": "defective"}], labels=["normal", "defective"]), _PromptAwareModel("defective")),
            (DocVQABenchmark, _QADataset("docvqa", [{"image": _square("white"), "question": "What is the invoice number?", "answers": ["A123"]}]), _PromptAwareModel("A123")),
            (DFDCBenchmark, _BaseStubDataset("dfdc", [{"frames": _frames(), "label_text": "fake"}], labels=["real", "fake"]), _PromptAwareModel("fake")),
            (TextCapsBenchmark, _CaptionDataset("textcaps", [{"image": _square("white"), "captions": ["Stop sign ahead", "A stop sign ahead"]}]), _PromptAwareModel("Stop sign ahead")),
            (LAION400MBenchmark, _CaptionDataset("laion400m", [{"image": _square("purple"), "captions": ["Purple square"]}]), _PromptAwareModel("Purple square")),
            (FairFaceBenchmark, _BaseStubDataset("fairface", [{"image": _square("pink"), "label": "adult"}], labels=["child", "adult"]), _PromptAwareModel("adult")),
            (LAION5BBenchmark, _CaptionDataset("laion5b", [{"image": _square("teal"), "captions": ["Teal square"]}]), _PromptAwareModel("Teal square")),
            (InternVidBenchmark, _CaptionDataset("internvid", [{"frames": _frames(), "captions": ["A person jumps"]}]), _PromptAwareModel("A person jumps")),
            (OpenVid1MBenchmark, _ChoiceDataset("openvid1m", [{"frames": _frames(), "question": "Which prompt matches the clip?", "choices": ["a person jumping", "a dog barking"], "answer": "a person jumping"}]), _PromptAwareModel("a person jumping")),
            (OpenImagesV4Benchmark, _BaseStubDataset("openimages_v4", [{"image": _square("lime"), "label": "car"}], labels=["car", "bus"]), _PromptAwareModel("car")),
            (MSCOCOCaptionBenchmark, _CaptionDataset("mscoco_caption", [{"image": _square("white"), "captions": ["A white square", "A bright square"]}]), _PromptAwareModel("A white square")),
        ]

        for benchmark_cls, dataset, model in cases:
            with self.subTest(benchmark=benchmark_cls.__name__):
                benchmark = benchmark_cls(dataset=dataset)
                report = benchmark.run(model=model, n=1, label_sample_size=2, show_progress=False)
                self.assertEqual(report["num_samples"], 1)
                self.assertTrue(report["results"][0]["correct"])

    def test_all_new_benchmarks_have_default_loader(self) -> None:
        loader_cases = [
            ("benchmarks.classification.lsun.LSUN", LSUNBenchmark),
            ("benchmarks.visual_qa.vqa_v2.VQAv2", VQAv2Benchmark),
            ("benchmarks.classification.fashion_mnist.FashionMNIST", FashionMNISTBenchmark),
            ("benchmarks.video_classification.kinetics.Kinetics", KineticsBenchmark),
            ("benchmarks.classification.places.Places", PlacesBenchmark),
            ("benchmarks.multiple_choice.conceptual_captions.ConceptualCaptions", ConceptualCaptionsBenchmark),
            ("benchmarks.visual_qa.gqa.GQA", GQABenchmark),
            ("benchmarks.classification.mvtec_ad.MVTecAD", MVTecADBenchmark),
            ("benchmarks.visual_qa.docvqa.DocVQA", DocVQABenchmark),
            ("benchmarks.video_classification.dfdc.DFDC", DFDCBenchmark),
            ("benchmarks.captioning.textcaps.TextCaps", TextCapsBenchmark),
            ("benchmarks.captioning.laion400m.LAION400M", LAION400MBenchmark),
            ("benchmarks.classification.fairface.FairFace", FairFaceBenchmark),
            ("benchmarks.captioning.laion5b.LAION5B", LAION5BBenchmark),
            ("benchmarks.captioning.internvid.InternVid", InternVidBenchmark),
            ("benchmarks.multiple_choice.openvid1m.OpenVid1M", OpenVid1MBenchmark),
            ("benchmarks.classification.openimages_v4.OpenImagesV4", OpenImagesV4Benchmark),
            ("benchmarks.captioning.mscoco_caption.MSCOCOCaption", MSCOCOCaptionBenchmark),
        ]

        for target, benchmark_cls in loader_cases:
            del target
            fake_dataset = _BaseStubDataset("fake", [{"image": _square("white"), "label": "label"}], labels=["label"])
            with self.subTest(benchmark=benchmark_cls.__name__):
                with patch.object(benchmark_cls, "dataset_cls", return_value=fake_dataset) as loader_cls:
                    benchmark = benchmark_cls()
                    self.assertIs(benchmark.dataset, fake_dataset)
                    loader_cls.assert_called_once()

    def test_fashion_mnist_default_evaluation_uses_test_split(self) -> None:
        fake_dataset = _BaseStubDataset("fake", [{"image": _square("white"), "label": "label"}], labels=["label"])
        with patch.object(FashionMNISTBenchmark, "dataset_cls", return_value=fake_dataset) as loader_cls:
            benchmark = FashionMNISTBenchmark()
        self.assertIs(benchmark.dataset, fake_dataset)
        loader_cls.assert_called_once_with(split="test", streaming=True)

    def test_declared_caption_datasets_use_captioning_benchmarks(self) -> None:
        for benchmark_cls in (
            LAION400MBenchmark,
            LAION5BBenchmark,
            InternVidBenchmark,
        ):
            self.assertTrue(issubclass(benchmark_cls, CaptioningBenchmark), benchmark_cls.__name__)

    def test_declared_caption_datasets_use_caption_dataset_bases(self) -> None:
        self.assertTrue(issubclass(LAION400M, HFCaptionDataset))
        self.assertTrue(issubclass(LAION5B, HFCaptionDataset))
        self.assertTrue(issubclass(HDTF, HFVideoCaptionDataset))
        self.assertTrue(issubclass(InternVid, HFVideoCaptionDataset))

        dataset = HDTF.__new__(HDTF)
        dataset.frame_keys = ("frames",)
        dataset.max_frames = 4
        representative_frame = dataset.get_image_from_row({"frames": _frames()})
        self.assertEqual(representative_frame.size, (24, 24))

    def test_fashion_mnist_normalizes_equivalent_tshirt_spelling(self) -> None:
        dataset = FashionMNIST.__new__(FashionMNIST)
        self.assertEqual(
            dataset.normalize_text("T - shirt / top"),
            dataset.normalize_text("T-shirt/top"),
        )


if __name__ == "__main__":
    unittest.main()
