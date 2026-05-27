"""Prepare held-out-safe Fashion-MNIST data for GPT-4o vision fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import FashionMNIST

from build_vision_jsonl import build_jsonl


SYSTEM_PROMPT = "Classify Fashion-MNIST clothing images using exactly one allowed label."


def _prompt(labels: list[str]) -> str:
    label_set = ", ".join(labels)
    return (
        "Return exactly ONE label from this list (one item only, no extra words):\n"
        f"{label_set}"
    )


def _export_records(
    *,
    rows: list[dict],
    prefix: str,
    labels: list[str],
    output_dir: Path,
) -> Path:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{prefix}_manifest.jsonl"
    prompt = _prompt(labels)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for index, row in enumerate(rows):
            image_path = image_dir / f"{prefix}_{index:05d}.png"
            row["image"].convert("RGB").save(image_path, format="PNG")
            label_index = int(row["label"])
            record = {
                "image_path": image_path.relative_to(output_dir).as_posix(),
                "prompt": prompt,
                "answer": labels[label_index],
                "detail": "low",
            }
            manifest.write(json.dumps(record, ensure_ascii=True) + "\n")
    return manifest_path


def select_balanced_indices(
    rows: list[dict],
    *,
    label_count: int,
    train_per_class: int,
    validation_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    grouped: dict[int, list[int]] = {label: [] for label in range(label_count)}
    for index, row in enumerate(rows):
        label = int(row["label"])
        if label in grouped:
            grouped[label].append(index)

    required = train_per_class + validation_per_class
    for label, indices in grouped.items():
        if len(indices) < required:
            raise ValueError(
                f"label {label} has {len(indices)} records; {required} are required"
            )
        rng.shuffle(indices)

    train_indices = [
        index
        for label in range(label_count)
        for index in grouped[label][:train_per_class]
    ]
    validation_indices = [
        index
        for label in range(label_count)
        for index in grouped[label][train_per_class:required]
    ]
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=30)
    parser.add_argument("--validation-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine-tuning/data/fashion_mnist"),
    )
    args = parser.parse_args()
    if args.train_per_class < 1:
        raise SystemExit("--train-per-class must be at least 1.")
    if args.validation_per_class < 1:
        raise SystemExit("--validation-per-class must be at least 1.")

    dataset = FashionMNIST(split="train", streaming=False)
    labels = dataset.get_labels([])
    if not labels:
        raise SystemExit("Fashion-MNIST labels could not be loaded.")

    source_rows = [dict(row) for row in dataset.ds]
    train_indices, validation_indices = select_balanced_indices(
        source_rows,
        label_count=len(labels),
        train_per_class=args.train_per_class,
        validation_per_class=args.validation_per_class,
        seed=args.seed,
    )
    training_rows = [source_rows[index] for index in train_indices]
    validation_rows = [source_rows[index] for index in validation_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = _export_records(
        rows=training_rows,
        prefix="train",
        labels=labels,
        output_dir=args.output_dir,
    )
    validation_manifest = _export_records(
        rows=validation_rows,
        prefix="validation",
        labels=labels,
        output_dir=args.output_dir,
    )
    train_output = args.output_dir / "train_openai.jsonl"
    validation_output = args.output_dir / "validation_openai.jsonl"
    build_jsonl(train_manifest, train_output, args.output_dir, SYSTEM_PROMPT)
    build_jsonl(
        validation_manifest, validation_output, args.output_dir, SYSTEM_PROMPT
    )

    metadata = {
        "dataset": "zalando-datasets/fashion_mnist",
        "training_source_split": "train",
        "evaluation_split": "test",
        "seed": args.seed,
        "train_examples": len(training_rows),
        "validation_examples": len(validation_rows),
        "train_per_class": args.train_per_class,
        "validation_per_class": args.validation_per_class,
        "train_class_counts": dict(
            Counter(labels[int(row["label"])] for row in training_rows)
        ),
        "validation_class_counts": dict(
            Counter(labels[int(row["label"])] for row in validation_rows)
        ),
        "labels": labels,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(training_rows)} balanced training examples to {train_output}")
    print(
        f"Wrote {len(validation_rows)} balanced validation examples to "
        f"{validation_output}"
    )
    print("Reserved Fashion-MNIST test split for benchmark evaluation.")


if __name__ == "__main__":
    main()
