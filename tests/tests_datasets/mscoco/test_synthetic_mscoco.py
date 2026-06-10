from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

from benchmarks._base_benchmark import BaseBenchmark
from benchmarks import MSCOCOBenchmark
from dataset._base_dataset import BaseDataset
from models import GPT4, Llava


FIXTURE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = FIXTURE_DIR / "images"
ANNOTATIONS_PATH = FIXTURE_DIR / "instances_val2017.json"


class _SyntheticMSCOCODataset(BaseDataset):
    def __init__(self, annotations_path: Path, image_dir: Path) -> None:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        self.name = "mscoco"
        self.labels = ["sports ball"]
        self.category_id_to_label = {37: "sports ball"}
        self._image_dir = image_dir
        self._rows = self._build_rows(payload)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self._rows)

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self._rows[:n]

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image_path = row["image"]["path"]
        return Image.open(image_path).convert("RGB")

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        labels = {
            self.category_id_to_label[item["category_id"]]
            for item in row.get("annotations", [])
            if item.get("category_id") in self.category_id_to_label
        }
        return sorted(labels)

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in row.get("annotations", []):
            bbox = item.get("bbox")
            category_id = item.get("category_id")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            out.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "label": self.category_id_to_label.get(category_id, ""),
                    "category_id": category_id,
                }
            )
        return out

    def _build_rows(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        anns_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
        for ann in payload.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            image_id = ann.get("image_id")
            if isinstance(image_id, int):
                anns_by_image_id.setdefault(image_id, []).append(ann)

        rows: List[Dict[str, Any]] = []
        for image_info in payload.get("images", []):
            if not isinstance(image_info, dict):
                continue
            image_id = image_info.get("id")
            file_name = image_info.get("file_name")
            if not isinstance(image_id, int) or not isinstance(file_name, str):
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "width": image_info.get("width"),
                    "height": image_info.get("height"),
                    "image": {"path": str(self._image_dir / file_name)},
                    "annotations": list(anns_by_image_id.get(image_id, [])),
                }
            )
        return rows


class _DotDetectorModel:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def predict(self, image: Image.Image, prompt: str) -> str:
        rgb = image.convert("RGB")
        width, height = rgb.size
        background = rgb.getpixel((0, 0))
        xs: List[int] = []
        ys: List[int] = []
        for y in range(height):
            for x in range(width):
                if rgb.getpixel((x, y)) != background:
                    xs.append(x)
                    ys.append(y)

        self.calls.append({"mode": rgb.mode, "size": rgb.size, "prompt": prompt})
        if not xs or not ys:
            return ""

        x0 = min(xs) / width
        y0 = min(ys) / height
        box_w = (max(xs) - min(xs)) / width
        box_h = (max(ys) - min(ys)) / height
        return f"sports ball: [{x0:.6f}, {y0:.6f}, {box_w:.6f}, {box_h:.6f}]"


class _SyntheticMSCOCOBenchmark(MSCOCOBenchmark):
    def __init__(self, dataset: _SyntheticMSCOCODataset) -> None:
        BaseBenchmark.__init__(self, dataset=dataset, name="mscoco")


class SyntheticMSCOCOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _SyntheticMSCOCODataset(
            annotations_path=ANNOTATIONS_PATH,
            image_dir=IMAGE_DIR,
        )
        self.benchmark = _SyntheticMSCOCOBenchmark(dataset=self.dataset)

    def test_synthetic_coco_images_run_through_benchmark(self) -> None:
        model = _DotDetectorModel()

        report = self.benchmark.run(
            model=model,
            n=3,
            label_sample_size=3,
            show_progress=False,
        )

        self.assertEqual(report["benchmark"], "mscoco")
        self.assertEqual(report["dataset"], "mscoco")
        self.assertEqual(report["num_samples"], 3)
        self.assertEqual(len(model.calls), 3)

        for call in model.calls:
            self.assertEqual(call["mode"], "RGB")
            self.assertIn("USER: <image>", call["prompt"])
            self.assertIn("sports ball", call["prompt"])

        for result in report["results"]:
            self.assertTrue(result["correct"])
            self.assertEqual(result["valid_labels"], ["sports ball"])
            self.assertEqual(len(result["ground_truth_boxes"]), 1)
            self.assertEqual(len(result["predicted_boxes"]), 1)
            self.assertGreaterEqual(result["f1"], 1.0)
            self.assertGreaterEqual(result["mean_iou_matched"], 0.95)

    def test_synthetic_coco_images_run_through_configured_model(self) -> None:
        model = self._make_configured_model()
        report = self.benchmark.run(
            model=model,
            n=3,
            label_sample_size=3,
            show_progress=False,
        )

        self.assertEqual(report["benchmark"], "mscoco")
        self.assertEqual(report["dataset"], "mscoco")
        self.assertEqual(report["num_samples"], 3)
        self.assertEqual(len(report["results"]), 3)

        for result in report["results"]:
            self.assertEqual(result["valid_labels"], ["sports ball"])
            self.assertEqual(len(result["ground_truth_boxes"]), 1)
            self.assertGreaterEqual(len(result["predicted_boxes"]), 1)

    def _make_configured_model(self):
        enabled = os.getenv("MSCOCO_RUN_LIVE_MODEL_TEST", "").strip().lower()
        if enabled not in {"1", "true", "yes"}:
            self.skipTest("Set MSCOCO_RUN_LIVE_MODEL_TEST=1 to run the real model adapter test.")

        provider = os.getenv("MSCOCO_TEST_MODEL", "").strip().lower()
        if not provider:
            if os.getenv("OPENAI_API_KEY"):
                provider = "gpt4"
            else:
                self.skipTest("Set MSCOCO_TEST_MODEL or OPENAI_API_KEY to run a real model adapter.")

        if provider == "gpt4":
            model_id = os.getenv("MSCOCO_TEST_OPENAI_MODEL", "gpt-4o")
            max_new_tokens = int(os.getenv("MSCOCO_TEST_MAX_NEW_TOKENS", "200"))
            return GPT4(model_id=model_id, max_new_tokens=max_new_tokens, temperature=0.0)

        if provider == "llava":
            model_id = os.getenv("MSCOCO_TEST_LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
            max_new_tokens = int(os.getenv("MSCOCO_TEST_MAX_NEW_TOKENS", "200"))
            return Llava(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                stream=False,
            )

        self.fail(f"Unsupported MSCOCO_TEST_MODEL value: {provider}")


if __name__ == "__main__":
    unittest.main()
