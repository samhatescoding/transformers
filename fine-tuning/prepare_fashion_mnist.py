"""Prepare held-out-safe Fashion-MNIST data for GPT-4o vision fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-examples", type=int, default=100)
    parser.add_argument("--validation-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine-tuning/data/fashion_mnist"),
    )
    args = parser.parse_args()
    if args.train_examples < 10:
        raise SystemExit("--train-examples must be at least 10.")
    if args.validation_examples < 1:
        raise SystemExit("--validation-examples must be at least 1.")

    total = args.train_examples + args.validation_examples
    dataset = FashionMNIST(split="train", streaming=False)
    labels = dataset.get_labels([])
    if not labels:
        raise SystemExit("Fashion-MNIST labels could not be loaded.")

    indices = list(range(len(dataset.ds)))
    random.Random(args.seed).shuffle(indices)
    selected = [dict(dataset.ds[index]) for index in indices[:total]]
    training_rows = selected[: args.train_examples]
    validation_rows = selected[args.train_examples :]

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
        "train_examples": args.train_examples,
        "validation_examples": args.validation_examples,
        "labels": labels,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.train_examples} training examples to {train_output}")
    print(f"Wrote {args.validation_examples} validation examples to {validation_output}")
    print("Reserved Fashion-MNIST test split for benchmark evaluation.")


if __name__ == "__main__":
    main()
