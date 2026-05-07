from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from benchmarks import (
    ConceptualCaptionsBenchmark,
    DFDCBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    MSCOCOCaptionBenchmark,
    GQABenchmark,
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    OpenVid1MBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    VQAv2Benchmark,
    Flickr30kEntitiesBenchmark,
)


class StubDataset:
    def __init__(self, name, rows, labels=None):
        self.name = name
        self._rows = rows
        self.labels = list(labels or [])

    def __iter__(self):
        return iter(self._rows)

    def get_labels(self, rows):
        del rows
        return list(self.labels)

    def get_samples(self, n):
        return self._rows[:n]

    def get_image_from_row(self, row):
        if row.get("image") is not None:
            return row["image"]
        if row.get("frames"):
            return row["frames"][0]
        if row.get("source_image") is not None:
            return row["source_image"]
        raise KeyError("row does not contain image data")

    def get_labels_img(self, row):
        label = row.get("label_text", row.get("label"))
        return [str(label)] if label is not None else []

    def get_question_from_row(self, row):
        return str(row.get("question", ""))

    def get_answers_from_row(self, row):
        answers = row.get("answers")
        if isinstance(answers, list):
            return list(answers)
        answer = row.get("answer")
        return [str(answer)] if answer is not None else []

    def get_choices_from_row(self, row):
        return [str(item) for item in row.get("choices", [])]

    def get_answer_from_row(self, row):
        return str(row.get("answer", ""))

    def get_captions_from_row(self, row):
        return list(row.get("captions", []))

    def normalize_text(self, text):
        text = str(text).strip().lower()
        for ch in [".", ",", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}"]:
            text = text.replace(ch, "")
        return " ".join(text.split())


class StubModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.name = "stub_model"

    def predict(self, image, prompt):
        del image
        del prompt
        return self.prediction


def square(color):
    return Image.new("RGB", (24, 24), color)


def frames():
    return [square("red"), square("green"), square("blue")]


def repeat_rows(base_row, count=10):
    rows = []
    for idx in range(count):
        row = dict(base_row)
        row["id"] = f"sample-{idx + 1}"
        rows.append(row)
    return rows


def save_report(benchmark_name, report):
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stub_model_{benchmark_name}.json"
    payload = {
        "model": "stub_model",
        "benchmark": benchmark_name,
        "report": report,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cases = [
        (LSUNBenchmark, StubDataset("lsun", repeat_rows({"image": square("white"), "label": "bedroom"}), labels=["bedroom", "classroom"]), StubModel("bedroom")),
        (VQAv2Benchmark, StubDataset("vqav2", repeat_rows({"image": square("white"), "question": "What color is the square?", "answers": ["white"]})), StubModel("white")),
        (FashionMNISTBenchmark, StubDataset("fashion_mnist", repeat_rows({"image": square("gray"), "label": "coat"}), labels=["coat", "dress"]), StubModel("coat")),
        (KineticsBenchmark, StubDataset("kinetics", repeat_rows({"frames": frames(), "label_text": "jumping"}), labels=["jumping", "running"]), StubModel("jumping")),
        (PlacesBenchmark, StubDataset("places", repeat_rows({"image": square("yellow"), "label": "kitchen"}), labels=["kitchen", "forest"]), StubModel("kitchen")),
        (ConceptualCaptionsBenchmark, StubDataset("conceptual_captions", repeat_rows({"image": square("orange"), "question": "Which caption matches the image?", "choices": ["A cat on a mat", "A dog in a field"], "answer": "A dog in a field"})), StubModel("B")),
        (GQABenchmark, StubDataset("gqa", repeat_rows({"image": square("blue"), "question": "What color is the square?", "answers": ["blue"]})), StubModel("blue")),
        (MVTecADBenchmark, StubDataset("mvtec_ad", repeat_rows({"image": square("black"), "label": "defective"}), labels=["normal", "defective"]), StubModel("defective")),
        (DocVQABenchmark, StubDataset("docvqa", repeat_rows({"image": square("white"), "question": "What is the invoice number?", "answers": ["A123"]})), StubModel("A123")),
        (DFDCBenchmark, StubDataset("dfdc", repeat_rows({"frames": frames(), "label_text": "fake"}), labels=["real", "fake"]), StubModel("fake")),
        (TextCapsBenchmark, StubDataset("textcaps", repeat_rows({"image": square("white"), "captions": ["Stop sign ahead", "A stop sign ahead"]})), StubModel("Stop sign ahead")),
        (LAION400MBenchmark, StubDataset("laion400m", repeat_rows({"image": square("purple"), "question": "Which caption matches the image?", "choices": ["Purple square", "Green square"], "answer": "Purple square"})), StubModel("A")),
        (FairFaceBenchmark, StubDataset("fairface", repeat_rows({"image": square("pink"), "label": "adult"}), labels=["child", "adult"]), StubModel("adult")),
        (LAION5BBenchmark, StubDataset("laion5b", repeat_rows({"image": square("teal"), "question": "Which caption matches the image?", "choices": ["Teal square", "Brown square"], "answer": "Teal square"})), StubModel("A")),
        (InternVidBenchmark, StubDataset("internvid", repeat_rows({"frames": frames(), "question": "Which caption matches the clip?", "choices": ["A person jumps", "A person sleeps"], "answer": "A person jumps"})), StubModel("A")),
        (OpenVid1MBenchmark, StubDataset("openvid1m", repeat_rows({"frames": frames(), "question": "Which prompt matches the clip?", "choices": ["a person jumping", "a dog barking"], "answer": "a person jumping"})), StubModel("A")),
        (OpenImagesV4Benchmark, StubDataset("openimages_v4", repeat_rows({"image": square("lime"), "label": "car"}), labels=["car", "bus"]), StubModel("car")),
        (OpenImagesV4DetectionBenchmark, StubDataset("openimages_v4_detection", repeat_rows({"image": square("white"), "annotations": [{"label": "car", "bbox": [1, 1, 8, 8]}]}), labels=["car"]), StubModel("car: [0.04, 0.04, 0.29, 0.29]")),
        (MSCOCOCaptionBenchmark, StubDataset("mscoco_caption", repeat_rows({"image": square("white"), "captions": ["A white square", "A bright white square"]})), StubModel("A white square")),
        (Flickr30kEntitiesBenchmark, StubDataset("flickr30k_entities", repeat_rows({"image": square("white"), "annotations": [{"label": "helmet", "bbox": [1, 1, 8, 8]}]}), labels=["helmet"]), StubModel("helmet: [0.04, 0.04, 0.29, 0.29]")),
    ]

    for benchmark_cls, dataset, model in cases:
        benchmark = benchmark_cls(dataset=dataset)
        report = benchmark.run(model=model, n=10, label_sample_size=10, show_progress=False)
        save_report(benchmark.name, report)


if __name__ == "__main__":
    main()
