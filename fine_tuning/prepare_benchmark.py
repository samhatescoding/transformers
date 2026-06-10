"""Export vision fine-tuning manifests from any repository benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.captioning import CaptioningBenchmark
from benchmarks.detection import DetectionBenchmark

from benchmark_registry import BENCHMARK_CLASSES


SYSTEM_PROMPT = "Follow the visual task instructions and return only the requested answer format."


def prompt_for_training(prompt: str) -> str:
    return (
        str(prompt)
        .replace("USER:", "")
        .replace("ASSISTANT:", "")
        .replace("<image>", "")
        .strip()
    )


def detection_answer(
    benchmark: DetectionBenchmark,
    row: dict[str, Any],
    image: Image.Image,
) -> str:
    boxes = benchmark.postprocess_ground_truth_boxes(
        benchmark.get_ground_truth_boxes_for_row(row),
        image=image,
    )
    if not boxes:
        raise ValueError("row has no usable detection annotations")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("row image has invalid dimensions")
    lines: list[str] = []
    for box in boxes:
        label = str(box.get("label", "")).strip()
        xyxy = box.get("xyxy")
        if not label or not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in xyxy]
        normalized = [
            x0 / width,
            y0 / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
        ]
        coords = ", ".join(f"{max(0.0, min(1.0, value)):.6f}" for value in normalized)
        lines.append(f"{label}: [{coords}]")
    if not lines:
        raise ValueError("row has no labeled detection boxes")
    return "\n".join(lines)


def answer_for_row(benchmark: Any, row: dict[str, Any], image: Image.Image) -> str:
    if isinstance(benchmark, DetectionBenchmark):
        return detection_answer(benchmark, row, image)
    if isinstance(benchmark, CaptioningBenchmark):
        answers = [str(answer).strip() for answer in benchmark._get_captions(row)]
    else:
        answers = [str(answer).strip() for answer in benchmark.get_valid_labels_for_row(row)]
    answers = [answer for answer in answers if answer]
    if not answers:
        raise ValueError("row has no supervised answer")
    return answers[0]


def export_examples(
    *,
    benchmark: Any,
    count: int,
    label_sample_size: int,
    skip_examples: int = 0,
) -> list[dict[str, Any]]:
    requested_count = count + skip_examples
    rows, labels = benchmark.prepare(
        n=requested_count,
        label_sample_size=max(requested_count, label_sample_size),
    )
    if len(rows) < requested_count:
        raise ValueError(
            f"{benchmark.name} produced {len(rows)} examples; {requested_count} were requested"
        )
    rows = rows[skip_examples:]
    records: list[dict[str, Any]] = []
    for row in rows:
        image = benchmark.get_image_for_row(row)
        if not isinstance(image, Image.Image):
            image = benchmark._coerce_image(image)
        image = image.convert("RGB")
        prompt_labels = benchmark.get_prompt_labels_for_row(row=row, labels=labels)
        prompt = prompt_for_training(
            benchmark.make_prompt(labels=prompt_labels, row=row, image=image)
        )
        records.append(
            {
                "image": image,
                "prompt": prompt,
                "answer": answer_for_row(benchmark, row, image),
            }
        )
    return records


def write_manifest(
    *,
    records: list[dict[str, Any]],
    prefix: str,
    output_dir: Path,
) -> Path:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{prefix}_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for index, record in enumerate(records):
            image_path = image_dir / f"{prefix}_{index:05d}.png"
            record["image"].save(image_path, format="PNG")
            payload = {
                "image_path": image_path.relative_to(output_dir).as_posix(),
                "prompt": record["prompt"],
                "answer": record["answer"],
                "system": SYSTEM_PROMPT,
            }
            manifest.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return manifest_path


def build_records(
    *,
    benchmark_cls: type,
    train_split: str,
    validation_split: str | None,
    train_examples: int,
    validation_examples: int,
    label_sample_size: int,
    streaming: bool,
    skip_examples: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_split is None or validation_split == train_split:
        benchmark = benchmark_cls(split=train_split, streaming=streaming)
        combined = export_examples(
            benchmark=benchmark,
            count=train_examples + validation_examples,
            label_sample_size=label_sample_size,
            skip_examples=skip_examples,
        )
        return combined[:train_examples], combined[train_examples:]
    train_records = export_examples(
        benchmark=benchmark_cls(split=train_split, streaming=streaming),
        count=train_examples,
        label_sample_size=label_sample_size,
        skip_examples=skip_examples,
    )
    validation_records = export_examples(
        benchmark=benchmark_cls(split=validation_split, streaming=streaming),
        count=validation_examples,
        label_sample_size=label_sample_size,
    )
    return train_records, validation_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_CLASSES))
    parser.add_argument("--train-examples", type=int, default=300)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument(
        "--train-split",
        default="train",
        help="Source split for training examples; do not use an evaluation split.",
    )
    parser.add_argument(
        "--validation-split",
        help="Optional validation source split. By default it is held out from --train-split.",
    )
    parser.add_argument("--label-sample-size", type=int, default=512)
    parser.add_argument(
        "--skip-examples",
        type=int,
        default=0,
        help="Reserve this many examples at the start of the training split for evaluation.",
    )
    parser.add_argument("--non-streaming", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.train_examples < 1 or args.validation_examples < 1:
        raise SystemExit("--train-examples and --validation-examples must be at least 1.")
    if args.label_sample_size < 1:
        raise SystemExit("--label-sample-size must be at least 1.")
    if args.skip_examples < 0:
        raise SystemExit("--skip-examples cannot be negative.")

    benchmark_cls = BENCHMARK_CLASSES[args.benchmark]
    validation_source = args.validation_split or args.train_split
    train_records, validation_records = build_records(
        benchmark_cls=benchmark_cls,
        train_split=args.train_split,
        validation_split=args.validation_split,
        train_examples=args.train_examples,
        validation_examples=args.validation_examples,
        label_sample_size=args.label_sample_size,
        streaming=not args.non_streaming,
        skip_examples=args.skip_examples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = write_manifest(
        records=train_records, prefix="train", output_dir=args.output_dir
    )
    validation_manifest = write_manifest(
        records=validation_records, prefix="validation", output_dir=args.output_dir
    )
    metadata = {
        "benchmark": args.benchmark,
        "benchmark_default_evaluation_split": benchmark_cls.default_split,
        "train_split": args.train_split,
        "validation_split": validation_source,
        "validation_is_held_out_slice": args.validation_split is None,
        "skipped_examples": args.skip_examples,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "system_prompt": SYSTEM_PROMPT,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(train_records)} training examples to {train_manifest}")
    print(f"Wrote {len(validation_records)} validation examples to {validation_manifest}")
    if benchmark_cls.default_split in {args.train_split, validation_source}:
        print(
            "WARNING: exported data uses this benchmark's default evaluation split; "
            "do not compare against the default evaluation run."
        )
    print("Evaluate on a distinct split or dataset slice not exported here.")


if __name__ == "__main__":
    main()
