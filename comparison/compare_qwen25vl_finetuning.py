from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_FILES = {
    "Baseline": "qwen2.5-vl-3b-base-test-split_fashion_mnist.json",
    "Original LoRA": "qwen2.5-vl-3b-fashion-mnist-lora_fashion_mnist.json",
    "Balanced LoRA": "qwen2.5-vl-3b-fashion-mnist-balanced-lora_fashion_mnist.json",
}

LABEL_ORDER = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

LABEL_ALIASES = {
    "t - shirt / top": "t-shirt/top",
    "t-shirt / top": "t-shirt/top",
    "t-shirt/top": "t-shirt/top",
}

MODEL_COLORS = {
    "Baseline": "#4c78a8",
    "Original LoRA": "#f58518",
    "Balanced LoRA": "#54a24b",
}


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return LABEL_ALIASES.get(label, label)


def load_result(path: Path, display_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload["report"]
    rows = []
    for item in report["results"]:
        valid_labels = item.get("valid_labels") or []
        if not valid_labels:
            raise ValueError(f"{path} sample {item.get('index')} has no valid label")
        stats = item.get("stats") or {}
        rows.append(
            {
                "model": display_name,
                "index": int(item["index"]),
                "target": normalize_label(valid_labels[0]),
                "prediction": normalize_label(item.get("prediction")),
                "correct": bool(item.get("correct")),
                "wall_clock_seconds": stats.get("wall_clock_time_seconds"),
                "generation_seconds": stats.get("generation_time_seconds"),
                "output_tokens": stats.get("output_tokens"),
            }
        )
    return pd.DataFrame(rows).sort_values("index"), dict(report.get("stats") or {})


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(baseline_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_delta(
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    count = len(baseline_correct)
    indices = rng.integers(0, count, size=(samples, count))
    deltas = (
        candidate_correct[indices].mean(axis=1)
        - baseline_correct[indices].mean(axis=1)
    )
    return {
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci025": float(np.quantile(deltas, 0.025)),
        "ci975": float(np.quantile(deltas, 0.975)),
        "probability_delta_above_zero": float((deltas > 0).mean()),
        "probability_delta_at_least_zero": float((deltas >= 0).mean()),
    }


def classification_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label in LABEL_ORDER:
        target = frame["target"] == label
        predicted = frame["prediction"] == label
        true_positive = int((target & predicted).sum())
        false_positive = int((~target & predicted).sum())
        false_negative = int((target & ~predicted).sum())
        support = int(target.sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "model": frame["model"].iloc[0],
                "label": label,
                "support": support,
                "predicted_count": int(predicted.sum()),
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def confusion_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)
    positions = {label: index for index, label in enumerate(LABEL_ORDER)}
    for row in frame.itertuples():
        if row.target in positions and row.prediction in positions:
            matrix[positions[row.target], positions[row.prediction]] += 1
    return matrix


def plot_accuracy(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(summary))
    values = summary["accuracy"].to_numpy()
    lower = values - summary["accuracy_ci_low"].to_numpy()
    upper = summary["accuracy_ci_high"].to_numpy() - values
    bars = ax.bar(
        x,
        values,
        yerr=np.vstack([lower, upper]),
        capsize=5,
        color=[MODEL_COLORS[name] for name in summary["model"]],
    )
    ax.set_xticks(x, summary["model"])
    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("Accuracy")
    ax.set_title("Fashion-MNIST Accuracy with 95% Wilson Intervals")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{row.correct}/{row.samples}\n({row.accuracy:.0%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=220)
    plt.close(fig)


def plot_per_class_recall(per_class: pd.DataFrame, output_dir: Path) -> None:
    pivot = per_class.pivot(index="label", columns="model", values="recall").reindex(
        LABEL_ORDER
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    width = 0.25
    x = np.arange(len(pivot))
    for offset, model in enumerate(MODEL_FILES):
        ax.bar(
            x + (offset - 1) * width,
            pivot[model],
            width=width,
            label=model,
            color=MODEL_COLORS[model],
        )
    ax.set_xticks(x, [label.replace("t-shirt/top", "T-shirt/top") for label in LABEL_ORDER])
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_recall.png", dpi=220)
    plt.close(fig)


def plot_confusions(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), sharex=True, sharey=True)
    maximum = max(confusion_matrix(frame).max() for frame in frames.values())
    image = None
    for ax, (model, frame) in zip(axes, frames.items()):
        matrix = confusion_matrix(frame)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_title(model)
        ax.set_xticks(range(len(LABEL_ORDER)), range(len(LABEL_ORDER)))
        ax.set_yticks(range(len(LABEL_ORDER)), range(len(LABEL_ORDER)))
        ax.set_xlabel("Predicted class index")
        for row in range(len(LABEL_ORDER)):
            for column in range(len(LABEL_ORDER)):
                if matrix[row, column]:
                    ax.text(
                        column,
                        row,
                        str(matrix[row, column]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[row, column] > maximum / 2 else "black",
                    )
    axes[0].set_ylabel("True class index")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8)
    fig.suptitle("Confusion Matrices (Class Indices Follow the Report Table)")
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.11, top=0.85, wspace=0.12)
    fig.savefig(output_dir / "confusion_matrices.png", dpi=220)
    plt.close(fig)


def plot_prediction_distribution(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    width = 0.25
    x = np.arange(len(LABEL_ORDER))
    for offset, (model, frame) in enumerate(frames.items()):
        counts = Counter(frame["prediction"])
        ax.bar(
            x + (offset - 1) * width,
            [counts[label] for label in LABEL_ORDER],
            width=width,
            label=model,
            color=MODEL_COLORS[model],
        )
    ax.set_xticks(x, [label.replace("t-shirt/top", "T-shirt/top") for label in LABEL_ORDER])
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Predictions")
    ax.set_title("Prediction Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_distribution.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/fine-tuning")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison/output/qwen25vl_finetuning"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    report_stats: dict[str, dict[str, Any]] = {}
    for model, filename in MODEL_FILES.items():
        path = args.results_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing result file: {path}")
        frames[model], report_stats[model] = load_result(path, model)

    baseline = frames["Baseline"]
    baseline_keys = list(zip(baseline["index"], baseline["target"]))
    for model, frame in frames.items():
        keys = list(zip(frame["index"], frame["target"]))
        if keys != baseline_keys:
            raise ValueError(f"{model} does not contain the same ordered evaluation samples")

    summary_rows = []
    per_class_rows = []
    for model, frame in frames.items():
        correct = int(frame["correct"].sum())
        total = len(frame)
        ci_low, ci_high = wilson_interval(correct, total)
        class_metrics = classification_rows(frame)
        per_class_rows.extend(class_metrics)
        macro_f1 = float(np.mean([row["f1"] for row in class_metrics]))
        macro_recall = float(np.mean([row["recall"] for row in class_metrics]))
        predictions = Counter(frame["prediction"])
        stats = report_stats[model]
        summary_rows.append(
            {
                "model": model,
                "samples": total,
                "correct": correct,
                "accuracy": correct / total,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "distinct_labels_predicted": len(predictions),
                "most_common_prediction": predictions.most_common(1)[0][0],
                "most_common_prediction_count": predictions.most_common(1)[0][1],
                "wall_clock_seconds": stats.get("wall_clock_time_seconds"),
                "seconds_per_sample": stats.get(
                    "wall_clock_time_per_sample_seconds_mean"
                ),
                "generation_seconds_per_sample": stats.get(
                    "total_generation_time_seconds_mean"
                ),
                "samples_per_second": stats.get("samples_per_second"),
                "tokens_per_second": stats.get("tokens_per_second"),
                "mean_output_tokens": stats.get("number_of_output_tokens_mean"),
                "model_load_seconds": stats.get("model_load_time_seconds"),
                "peak_cpu_ram_gib": (
                    stats.get("peak_cpu_ram_bytes") / (1024**3)
                    if stats.get("peak_cpu_ram_bytes") is not None
                    else None
                ),
                "peak_gpu_memory_gib": (
                    stats.get("peak_gpu_memory_bytes") / (1024**3)
                    if stats.get("peak_gpu_memory_bytes") is not None
                    else None
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    pairwise_rows = []
    transition_rows = []
    baseline_correct = baseline["correct"].to_numpy(dtype=float)
    baseline_errors = int((~baseline["correct"]).sum())
    for offset, model in enumerate(("Original LoRA", "Balanced LoRA")):
        candidate = frames[model]
        candidate_correct = candidate["correct"].to_numpy(dtype=float)
        baseline_only = int((baseline["correct"] & ~candidate["correct"]).sum())
        candidate_only = int((~baseline["correct"] & candidate["correct"]).sum())
        both_correct = int((baseline["correct"] & candidate["correct"]).sum())
        both_wrong = int((~baseline["correct"] & ~candidate["correct"]).sum())
        bootstrap = paired_bootstrap_delta(
            baseline_correct,
            candidate_correct,
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
        )
        candidate_errors = int((~candidate["correct"]).sum())
        pairwise_rows.append(
            {
                "comparison": f"{model} minus Baseline",
                "baseline_accuracy": float(baseline_correct.mean()),
                "candidate_accuracy": float(candidate_correct.mean()),
                "accuracy_delta": float(candidate_correct.mean() - baseline_correct.mean()),
                "relative_error_reduction": (
                    (baseline_errors - candidate_errors) / baseline_errors
                    if baseline_errors
                    else 0.0
                ),
                "both_correct": both_correct,
                "baseline_only_correct": baseline_only,
                "candidate_only_correct": candidate_only,
                "both_wrong": both_wrong,
                "mcnemar_exact_p": exact_mcnemar_p(baseline_only, candidate_only),
                **bootstrap,
            }
        )
        for base_row, candidate_row in zip(
            baseline.itertuples(), candidate.itertuples()
        ):
            if base_row.correct != candidate_row.correct:
                transition_rows.append(
                    {
                        "comparison": f"{model} versus Baseline",
                        "index": base_row.index,
                        "target": base_row.target,
                        "baseline_prediction": base_row.prediction,
                        "candidate_prediction": candidate_row.prediction,
                        "transition": (
                            "fixed"
                            if candidate_row.correct
                            else "regressed"
                        ),
                    }
                )

    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    per_class.to_csv(args.output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(
        args.output_dir / "paired_comparisons.csv", index=False
    )
    pd.DataFrame(transition_rows).to_csv(
        args.output_dir / "changed_outcomes.csv", index=False
    )
    pd.concat(frames.values(), ignore_index=True).to_csv(
        args.output_dir / "sample_predictions.csv", index=False
    )

    plot_accuracy(summary, args.output_dir)
    plot_per_class_recall(per_class, args.output_dir)
    plot_confusions(frames, args.output_dir)
    plot_prediction_distribution(frames, args.output_dir)

    print(summary.to_string(index=False))
    print()
    print(pd.DataFrame(pairwise_rows).to_string(index=False))


if __name__ == "__main__":
    main()
