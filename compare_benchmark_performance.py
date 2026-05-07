from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


BENCHMARK_FAMILY = {
    "blip3o_60k": "captioning",
    "conceptual_captions": "labeling",
    "conceptual_captions_caption": "captioning",
    "flickr30k": "captioning",
    "internvid": "captioning",
    "laion400m": "captioning",
    "laion5b": "captioning",
    "mscoco_caption": "captioning",
    "openvid1m": "captioning",
    "textcaps": "captioning",
    "docvqa": "qa",
    "gqa": "qa",
    "visual_cot": "qa",
    "visual_genome": "qa",
    "vqav2": "qa",
    "fairface": "labeling",
    "fashion_mnist": "labeling",
    "imagenet1k": "labeling",
    "inaturalist": "labeling",
    "kinetics": "labeling",
    "lsun": "labeling",
    "mvtec_ad": "labeling",
    "openimages_v4": "labeling",
    "places": "labeling",
    "ucf101": "labeling",
    "dfdc": "labeling",
    "flickr30k_entities": "detection",
    "lvis": "detection",
    "mscoco": "detection",
    "openimages_v4_detection": "detection",
}

PRIMARY_METRIC_BY_FAMILY = {
    "captioning": "mean_bleu",
    "qa": "accuracy",
    "labeling": "accuracy",
    "detection": "mean_f1",
}


def _load_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_summary.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("report", {})
        results = list(report.get("results", []))
        benchmark = str(payload.get("benchmark") or report.get("benchmark") or path.stem)
        family = BENCHMARK_FAMILY.get(benchmark, "other")
        accuracy = _mean(1.0 if item.get("correct") else 0.0 for item in results)
        mean_bleu = _mean(_coerce_float(item.get("bleu")) for item in results)
        mean_f1 = _mean(_coerce_float(item.get("f1")) for item in results)
        rows.append(
            {
                "file": path.name,
                "model": payload.get("model", "unknown"),
                "benchmark": benchmark,
                "family": family,
                "num_samples": len(results),
                "accuracy": accuracy,
                "mean_bleu": mean_bleu,
                "mean_f1": mean_f1,
                "primary_metric": PRIMARY_METRIC_BY_FAMILY.get(family, "accuracy"),
            }
        )
    return rows


def _mean(values) -> float:
    series = [value for value in values if value is not None]
    if not series:
        return 0.0
    return float(sum(series) / len(series))


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _make_chart(df: pd.DataFrame, output_dir: Path, family: str | None) -> None:
    chart_df = df.copy()
    if family:
        chart_df = chart_df[chart_df["family"] == family]
    if chart_df.empty:
        raise ValueError("No benchmark results matched the requested filter.")

    chart_df = chart_df.assign(score=chart_df.apply(lambda row: row[row["primary_metric"]], axis=1))
    chart_df = chart_df.sort_values(["family", "benchmark"])
    colors = {
        "captioning": "#c46b1a",
        "labeling": "#2f6db3",
        "qa": "#2f9b5f",
        "detection": "#8b4bb3",
        "other": "#6f6f6f",
    }

    fig, ax = plt.subplots(figsize=(max(10, len(chart_df) * 0.6), 6))
    ax.bar(
        chart_df["benchmark"],
        chart_df["score"],
        color=[colors.get(value, colors["other"]) for value in chart_df["family"]],
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    if family:
        ax.set_title(f"Model Performance Across {family.title()} Benchmarks")
    else:
        ax.set_title("Model Performance Across Benchmarks")
    ax.tick_params(axis="x", rotation=45, labelrotation=45)
    fig.tight_layout()

    suffix = family or "all"
    fig.savefig(output_dir / f"benchmark_performance_{suffix}.png", dpi=200)
    plt.close(fig)


def _write_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "benchmark_performance_by_benchmark.csv", index=False)

    family_summary = []
    for family, family_df in df.groupby("family"):
        if family_df.empty:
            continue
        metric = PRIMARY_METRIC_BY_FAMILY.get(family, "accuracy")
        family_summary.append(
            {
                "family": family,
                "primary_metric": metric,
                "mean_score": float(family_df[metric].mean()),
                "num_benchmarks": int(len(family_df)),
            }
        )
    pd.DataFrame(family_summary).sort_values("family").to_csv(
        output_dir / "benchmark_performance_by_family.csv",
        index=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare model performance across saved benchmark result files.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--family", choices=["captioning", "labeling", "qa", "detection"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(results_dir)
    if not rows:
        raise ValueError(f"No benchmark result files were found in {results_dir}")

    df = pd.DataFrame(rows)
    _write_summary_tables(df, output_dir)
    _make_chart(df, output_dir, family=args.family)
    if args.family is None:
        for family in ("captioning", "labeling", "qa", "detection"):
            if (df["family"] == family).any():
                _make_chart(df, output_dir, family=family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
