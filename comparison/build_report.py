from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCHMARK_FAMILY = {
    "blip3o_60k": "prompt_reconstruction",
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
    "hq_edit": "image_modification_vqa",
    "imgedit": "image_modification_vqa",
    "magicbrush": "image_modification_vqa",
    "sharegpt4o_image": "prompt_reconstruction",
    "pick_a_pic": "image_preference",
    "tad66k": "aesthetic_rating",
    "diffusiondb": "prompt_reconstruction",
}

PRIMARY_METRIC_BY_FAMILY = {
    "captioning": "mean_bleu",
    "qa": "accuracy",
    "labeling": "accuracy",
    "detection": "mean_f1",
    "image_modification_vqa": "accuracy",
    "image_preference": "accuracy",
    "aesthetic_rating": "aesthetic_score",
    "prompt_reconstruction": "accuracy",
}

FAMILY_ORDER = [
    "captioning",
    "qa",
    "labeling",
    "detection",
    "image_modification_vqa",
    "image_preference",
    "aesthetic_rating",
    "prompt_reconstruction",
    "other",
]
FAMILY_COLORS = {
    "captioning": "#c7681d",
    "qa": "#238b68",
    "labeling": "#2a6fbb",
    "detection": "#8a49b8",
    "image_modification_vqa": "#b33c86",
    "image_preference": "#d18f00",
    "aesthetic_rating": "#7a8f00",
    "prompt_reconstruction": "#008f95",
    "other": "#6e6e6e",
}
MODEL_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]


def _mean(values) -> float:
    series = [value for value in values if value is not None]
    if not series:
        return 0.0
    return float(sum(series) / len(series))


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_rows(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_summary.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("report", {})
        report_stats = report.get("stats", {}) if isinstance(report.get("stats", {}), dict) else {}
        results = list(report.get("results", []))
        benchmark = str(payload.get("benchmark") or report.get("benchmark") or path.stem)
        family = BENCHMARK_FAMILY.get(benchmark, "other")
        accuracy = _mean(1.0 if item.get("correct") else 0.0 for item in results)
        mean_bleu = _mean(_coerce_float(item.get("bleu")) for item in results)
        mean_f1 = _mean(_coerce_float(item.get("f1")) for item in results)
        mean_precision = _mean(_coerce_float(item.get("precision")) for item in results)
        mean_recall = _mean(_coerce_float(item.get("recall")) for item in results)
        mean_iou = _mean(_coerce_float(item.get("mean_iou_all_predictions")) for item in results)
        mean_absolute_error = _mean(_coerce_float(item.get("absolute_error")) for item in results)
        aesthetic_score = max(0.0, 1.0 - (mean_absolute_error / 9.0))
        primary_metric = PRIMARY_METRIC_BY_FAMILY.get(family, "accuracy")
        score = {
            "accuracy": accuracy,
            "mean_bleu": mean_bleu,
            "mean_f1": mean_f1,
            "aesthetic_score": aesthetic_score,
        }.get(primary_metric, accuracy)
        rows.append(
            {
                "file": path.name,
                "model": str(payload.get("model", "unknown")),
                "benchmark": benchmark,
                "family": family,
                "num_samples": len(results),
                "accuracy": accuracy,
                "mean_bleu": mean_bleu,
                "mean_f1": mean_f1,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_iou_all_predictions": mean_iou,
                "mean_absolute_error": mean_absolute_error,
                "aesthetic_score": aesthetic_score,
                "primary_metric": primary_metric,
                "score": score,
                "stats_source": str(report_stats.get("stats_source", "measured")),
                "stats_measured": bool(report_stats.get("stats_measured", True)),
            }
        )
    if not rows:
        raise ValueError(f"No benchmark result files were found in {results_dir}")
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    return df.sort_values(["family", "benchmark", "model"]).reset_index(drop=True)


def _load_telemetry_rows(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_summary.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("report", {})
        report_stats = report.get("stats", {}) if isinstance(report.get("stats", {}), dict) else {}
        benchmark = str(payload.get("benchmark") or report.get("benchmark") or path.stem)
        family = BENCHMARK_FAMILY.get(benchmark, "other")
        rows.append(
            {
                "file": path.name,
                "model": str(payload.get("model", "unknown")),
                "benchmark": benchmark,
                "family": family,
                "num_samples": int(report.get("num_samples", len(report.get("results", [])))),
                "stats_source": str(report_stats.get("stats_source", "measured")),
                "stats_measured": bool(report_stats.get("stats_measured", True)),
                "wall_clock_time_seconds": _coerce_float(report_stats.get("wall_clock_time_seconds")),
                "wall_clock_time_per_sample_seconds_mean": _coerce_float(report_stats.get("wall_clock_time_per_sample_seconds_mean")),
                "total_generation_time_seconds_mean": _coerce_float(report_stats.get("total_generation_time_seconds_mean")),
                "first_token_latency_seconds_mean": _coerce_float(report_stats.get("first_token_latency_seconds_mean")),
                "time_per_output_token_seconds_mean": _coerce_float(report_stats.get("time_per_output_token_seconds_mean")),
                "samples_per_second": _coerce_float(report_stats.get("samples_per_second")),
                "number_of_output_tokens_mean": _coerce_float(report_stats.get("number_of_output_tokens_mean")),
                "tokens_per_second": _coerce_float(report_stats.get("tokens_per_second")),
                "model_load_time_seconds": _coerce_float(report_stats.get("model_load_time_seconds")),
                "peak_cpu_ram_bytes": _coerce_float(report_stats.get("peak_cpu_ram_bytes")),
                "peak_gpu_memory_bytes": _coerce_float(report_stats.get("peak_gpu_memory_bytes")),
                "cpu_utilization_percent_mean": _coerce_float(report_stats.get("cpu_utilization_percent_mean")),
                "gpu_utilization_percent_mean": _coerce_float(report_stats.get("gpu_utilization_percent_mean")),
                "success_count": _coerce_float(report_stats.get("success_count")),
                "failure_count": _coerce_float(report_stats.get("failure_count")),
                "retry_count": _coerce_float(report_stats.get("retry_count")),
            }
        )
    if not rows:
        raise ValueError(f"No benchmark result files were found in {results_dir}")
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    return df.sort_values(["family", "benchmark", "model"]).reset_index(drop=True)


def _add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("benchmark")["score"]
    out["score_mean_for_benchmark"] = group.transform("mean")
    out["score_std_for_benchmark"] = group.transform("std").fillna(0.0)
    out["score_min_for_benchmark"] = group.transform("min")
    out["score_max_for_benchmark"] = group.transform("max")
    denom = (out["score_max_for_benchmark"] - out["score_min_for_benchmark"]).replace(0.0, np.nan)
    out["score_normalized_within_benchmark"] = ((out["score"] - out["score_min_for_benchmark"]) / denom).fillna(1.0)
    z_denom = out["score_std_for_benchmark"].replace(0.0, np.nan)
    out["score_z_within_benchmark"] = ((out["score"] - out["score_mean_for_benchmark"]) / z_denom).fillna(0.0)
    out["benchmark_rank"] = out.groupby("benchmark")["score"].rank(method="min", ascending=False)
    return out


def _build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("model", as_index=False)
        .agg(
            num_benchmarks=("benchmark", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            mean_normalized_score=("score_normalized_within_benchmark", "mean"),
            mean_z_score=("score_z_within_benchmark", "mean"),
            mean_rank=("benchmark_rank", "mean"),
            estimated_stats_benchmarks=("stats_measured", lambda s: int((~s.astype(bool)).sum())),
        )
        .fillna({"std_score": 0.0})
    )
    summary["first_place_finishes"] = summary["model"].map(
        df[df["benchmark_rank"] == 1].groupby("model").size().to_dict()
    ).fillna(0).astype(int)
    return summary.sort_values(["mean_normalized_score", "mean_score"], ascending=False).reset_index(drop=True)


def _build_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "family"], as_index=False, observed=True)
        .agg(
            num_benchmarks=("benchmark", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            mean_rank=("benchmark_rank", "mean"),
            mean_normalized_score=("score_normalized_within_benchmark", "mean"),
        )
        .sort_values(["family", "mean_normalized_score", "mean_score"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def _build_telemetry_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("model", as_index=False)
        .agg(
            num_benchmarks=("benchmark", "count"),
            mean_wall_clock_seconds=("wall_clock_time_per_sample_seconds_mean", "mean"),
            mean_generation_seconds=("total_generation_time_seconds_mean", "mean"),
            mean_model_load_seconds=("model_load_time_seconds", "mean"),
            mean_tokens_per_second=("tokens_per_second", "mean"),
            mean_output_tokens=("number_of_output_tokens_mean", "mean"),
            mean_samples_per_second=("samples_per_second", "mean"),
            peak_cpu_ram_bytes=("peak_cpu_ram_bytes", "max"),
            peak_gpu_memory_bytes=("peak_gpu_memory_bytes", "max"),
            mean_cpu_utilization_percent=("cpu_utilization_percent_mean", "mean"),
            measured_benchmarks=("stats_measured", lambda s: int(s.astype(bool).sum())),
            estimated_benchmarks=("stats_measured", lambda s: int((~s.astype(bool)).sum())),
        )
    )
    return summary.sort_values(["mean_wall_clock_seconds", "mean_generation_seconds"], ascending=[True, True]).reset_index(drop=True)


def _build_telemetry_benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["benchmark", "model"]).reset_index(drop=True)


def _build_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["benchmark", "family", "model", "score", "benchmark_rank", "primary_metric"]]
        .sort_values(["benchmark", "benchmark_rank", "model"])
        .reset_index(drop=True)
    )


def _build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = sorted(df["model"].unique())
    for model_a, model_b in combinations(models, 2):
        left = df[df["model"] == model_a][["benchmark", "score", "family"]].rename(columns={"score": "score_a"})
        right = df[df["model"] == model_b][["benchmark", "score"]].rename(columns={"score": "score_b"})
        merged = left.merge(right, on="benchmark", how="inner")
        if merged.empty:
            continue
        delta = merged["score_a"] - merged["score_b"]
        wins = int((delta > 1e-12).sum())
        losses = int((delta < -1e-12).sum())
        ties = int(len(merged) - wins - losses)
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "benchmarks_compared": int(len(merged)),
                "wins_a": wins,
                "wins_b": losses,
                "ties": ties,
                "win_rate_a": float((wins + 0.5 * ties) / len(merged)),
                "mean_delta_a_minus_b": float(delta.mean()),
                "median_delta_a_minus_b": float(delta.median()),
            }
        )
        rows.append(
            {
                "model_a": model_b,
                "model_b": model_a,
                "benchmarks_compared": int(len(merged)),
                "wins_a": losses,
                "wins_b": wins,
                "ties": ties,
                "win_rate_a": float((losses + 0.5 * ties) / len(merged)),
                "mean_delta_a_minus_b": float((-delta).mean()),
                "median_delta_a_minus_b": float((-delta).median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_a", "model_b"]).reset_index(drop=True)


def _build_bootstrap(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    pivot = df.pivot_table(index="benchmark", columns="model", values="score", aggfunc="mean")
    benchmarks = pivot.index.to_numpy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for model in pivot.columns:
        observed = pivot[model].dropna()
        if observed.empty:
            continue
        benchmark_names = observed.index.to_numpy()
        values = observed.to_numpy(dtype=float)
        sample_count = len(values)
        draws = rng.integers(0, sample_count, size=(n_samples, sample_count))
        sampled_means = values[draws].mean(axis=1)
        rows.append(
            {
                "model": model,
                "benchmarks_used": int(sample_count),
                "observed_mean_score": float(values.mean()),
                "bootstrap_mean_score": float(sampled_means.mean()),
                "bootstrap_std": float(sampled_means.std()),
                "ci05": float(np.quantile(sampled_means, 0.05)),
                "ci25": float(np.quantile(sampled_means, 0.25)),
                "ci50": float(np.quantile(sampled_means, 0.50)),
                "ci75": float(np.quantile(sampled_means, 0.75)),
                "ci95": float(np.quantile(sampled_means, 0.95)),
            }
        )
    del benchmarks
    return pd.DataFrame(rows).sort_values("observed_mean_score", ascending=False).reset_index(drop=True)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _plot_model_mean_scores(model_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(model_summary["model"], model_summary["mean_score"], color=MODEL_COLORS[: len(model_summary)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean Primary Score")
    ax.set_title("Average Score Across Benchmarks")
    fig.tight_layout()
    fig.savefig(output_dir / "model_mean_scores.png", dpi=220)
    plt.close(fig)


def _plot_telemetry_bar(
    df: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = df.dropna(subset=[value_col]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["model"], plot_df[value_col], color=MODEL_COLORS[: len(plot_df)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def _plot_telemetry_by_benchmark(
    df: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = df.dropna(subset=[value_col]).copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="benchmark", columns="model", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))
    pivot.plot(kind="bar", ax=ax, color=MODEL_COLORS[: len(pivot.columns)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35, labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def _plot_stats_coverage(telemetry_summary: pd.DataFrame, output_dir: Path) -> None:
    if telemetry_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(telemetry_summary))
    ax.bar(x, telemetry_summary["measured_benchmarks"], label="Measured", color="#2a6fbb")
    ax.bar(
        x,
        telemetry_summary["estimated_benchmarks"],
        bottom=telemetry_summary["measured_benchmarks"],
        label="Estimated",
        color="#c7681d",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(telemetry_summary["model"])
    ax.set_ylabel("Benchmarks")
    ax.set_title("Telemetry Coverage by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "telemetry_coverage.png", dpi=220)
    plt.close(fig)


def _plot_model_normalized_scores(model_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(model_summary["model"], model_summary["mean_normalized_score"], color=MODEL_COLORS[: len(model_summary)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean Benchmark-Normalized Score")
    ax.set_title("Average Relative Performance Within Each Benchmark")
    fig.tight_layout()
    fig.savefig(output_dir / "model_normalized_scores.png", dpi=220)
    plt.close(fig)


def _plot_family_grouped_bars(family_summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = family_summary.pivot(index="family", columns="model", values="mean_score").reindex(FAMILY_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, color=MODEL_COLORS[: len(pivot.columns)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean Score")
    ax.set_title("Model Performance by Benchmark Family")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_dir / "family_grouped_bars.png", dpi=220)
    plt.close(fig)


def _plot_score_heatmap(df: pd.DataFrame, output_dir: Path, normalized: bool) -> None:
    value_col = "score_normalized_within_benchmark" if normalized else "score"
    title = "Benchmark Comparison Heatmap (Normalized Within Benchmark)" if normalized else "Benchmark Comparison Heatmap (Raw Primary Score)"
    name = "score_heatmap_normalized.png" if normalized else "score_heatmap_raw.png"
    pivot = df.pivot(index="benchmark", columns="model", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0 if normalized else max(1.0, float(np.nanmax(pivot.to_numpy()))))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=220)
    plt.close(fig)


def _plot_rank_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot(index="benchmark", columns="model", values="benchmark_rank").sort_index()
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    vmax = max(1, int(np.nanmax(pivot.to_numpy())))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="plasma_r", vmin=1, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Benchmark Rank Heatmap (1 = Best)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_heatmap.png", dpi=220)
    plt.close(fig)


def _plot_pairwise_heatmap(pairwise: pd.DataFrame, output_dir: Path) -> None:
    pivot = pairwise.pivot(index="model_a", columns="model_b", values="win_rate_a").sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Pairwise Win-Rate Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_win_rate_heatmap.png", dpi=220)
    plt.close(fig)


def _plot_distribution_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    models = list(sorted(df["model"].unique()))
    series = [df.loc[df["model"] == model, "score"].to_numpy(dtype=float) for model in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(series, tick_labels=models, patch_artist=True)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Benchmark Score")
    ax.set_title("Distribution of Benchmark Scores by Model")
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_boxplot.png", dpi=220)
    plt.close(fig)


def _plot_benchmark_spread(df: pd.DataFrame, output_dir: Path) -> None:
    spread = (
        df.groupby(["benchmark", "family"], as_index=False)
        .agg(
            best_score=("score", "max"),
            worst_score=("score", "min"),
            mean_score=("score", "mean"),
        )
    )
    spread["spread"] = spread["best_score"] - spread["worst_score"]
    spread = spread.sort_values("spread", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        spread["benchmark"],
        spread["spread"],
        color=[FAMILY_COLORS.get(str(family), FAMILY_COLORS["other"]) for family in spread["family"]],
    )
    ax.set_ylabel("Best - Worst Score")
    ax.set_title("Benchmark Difficulty Spread Across Models")
    ax.tick_params(axis="x", rotation=45, labelrotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_spread.png", dpi=220)
    plt.close(fig)


def _plot_first_place_finishes(model_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(model_summary["model"], model_summary["first_place_finishes"], color=MODEL_COLORS[: len(model_summary)])
    ax.set_ylabel("Benchmark Wins")
    ax.set_title("First-Place Finishes by Model")
    fig.tight_layout()
    fig.savefig(output_dir / "first_place_finishes.png", dpi=220)
    plt.close(fig)


def _plot_bootstrap_intervals(bootstrap_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = bootstrap_df.sort_values("observed_mean_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    x = ordered["observed_mean_score"].to_numpy(dtype=float)
    left = x - ordered["ci05"].to_numpy(dtype=float)
    right = ordered["ci95"].to_numpy(dtype=float) - x
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, xerr=[left, right], fmt="o", color="#1f1f1f", ecolor="#4f79c8", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Mean Score")
    ax.set_title("Bootstrap Confidence Intervals Across Benchmarks")
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_confidence_intervals.png", dpi=220)
    plt.close(fig)


def _plot_radar(family_summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = family_summary.pivot(index="model", columns="family", values="mean_score").reindex(columns=[f for f in FAMILY_ORDER if f != "other"]).fillna(0.0)
    categories = list(pivot.columns)
    if not categories:
        return
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for idx, (model, row) in enumerate(pivot.iterrows()):
        values = row.to_list()
        values += values[:1]
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(angles, values, linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Family Radar Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    fig.tight_layout()
    fig.savefig(output_dir / "family_radar.png", dpi=220)
    plt.close(fig)


def _plot_family_small_multiples(df: pd.DataFrame, output_dir: Path) -> None:
    families = [family for family in FAMILY_ORDER if family != "other" and (df["family"] == family).any()]
    if not families:
        return
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 3.5 * len(families)), squeeze=False)
    for ax, family in zip(axes[:, 0], families):
        family_df = df[df["family"] == family].sort_values(["benchmark", "model"])
        pivot = family_df.pivot(index="benchmark", columns="model", values="score")
        pivot.plot(kind="bar", ax=ax, color=MODEL_COLORS[: len(pivot.columns)])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{family.title()} Benchmarks")
        ax.tick_params(axis="x", rotation=35, labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "family_small_multiples.png", dpi=220)
    plt.close(fig)


def _write_markdown_summary(
    benchmark_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    telemetry_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_model = model_summary.iloc[0]
    lines = [
        "# Comparison Summary",
        "",
        f"- Best overall mean benchmark-normalized score: `{best_model['model']}` ({best_model['mean_normalized_score']:.3f})",
        f"- Best raw mean score: `{model_summary.sort_values('mean_score', ascending=False).iloc[0]['model']}`",
        f"- Most first-place finishes: `{model_summary.sort_values('first_place_finishes', ascending=False).iloc[0]['model']}`",
        f"- Benchmarks with estimated telemetry: `{int((~benchmark_df['stats_measured']).sum())}` of `{len(benchmark_df)}`",
        "",
        "## Models",
        "",
    ]
    for _, row in model_summary.iterrows():
        lines.append(
            f"- `{row['model']}`: mean_score={row['mean_score']:.3f}, normalized={row['mean_normalized_score']:.3f}, "
            f"mean_rank={row['mean_rank']:.2f}, wins={int(row['first_place_finishes'])}, "
            f"estimated_stats_benchmarks={int(row['estimated_stats_benchmarks'])}"
        )
    lines.extend(["", "## Strongest Family Leaders", ""])
    family_leaders = family_summary.sort_values(["family", "mean_normalized_score"], ascending=[True, False]).groupby("family").head(1)
    for _, row in family_leaders.iterrows():
        lines.append(f"- `{row['family']}`: `{row['model']}` with mean_score={row['mean_score']:.3f}")
    if not pairwise.empty:
        lines.extend(["", "## Pairwise Edges", ""])
        best_pairwise = pairwise.sort_values(["win_rate_a", "mean_delta_a_minus_b"], ascending=False).head(5)
        for _, row in best_pairwise.iterrows():
            lines.append(
                f"- `{row['model_a']}` vs `{row['model_b']}`: win_rate={row['win_rate_a']:.3f}, "
                f"mean_delta={row['mean_delta_a_minus_b']:.3f} across {int(row['benchmarks_compared'])} benchmarks"
            )
    if not bootstrap_df.empty:
        lines.extend(["", "## Bootstrap Intervals", ""])
        for _, row in bootstrap_df.iterrows():
            lines.append(
                f"- `{row['model']}`: observed={row['observed_mean_score']:.3f}, "
                f"90% CI=[{row['ci05']:.3f}, {row['ci95']:.3f}]"
            )
    if not telemetry_summary.empty:
        lines.extend(["", "## Telemetry", ""])
        for _, row in telemetry_summary.iterrows():
            lines.append(
                f"- `{row['model']}`: mean_wall_clock={row['mean_wall_clock_seconds']:.2f}s, "
                f"mean_generation={row['mean_generation_seconds']:.2f}s, "
                f"peak_cpu_ram={row['peak_cpu_ram_bytes'] / (1024 ** 3):.2f} GiB, "
                f"measured={int(row['measured_benchmarks'])}, estimated={int(row['estimated_benchmarks'])}"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(results_dir: Path, output_dir: Path, bootstrap_samples: int, seed: int) -> None:
    _ensure_output_dir(output_dir)
    benchmark_df = _add_relative_metrics(_load_rows(results_dir))
    telemetry_df = _load_telemetry_rows(results_dir)
    model_summary = _build_model_summary(benchmark_df)
    family_summary = _build_family_summary(benchmark_df)
    ranking_table = _build_ranking_table(benchmark_df)
    pairwise = _build_pairwise(benchmark_df)
    bootstrap_df = _build_bootstrap(benchmark_df, n_samples=bootstrap_samples, seed=seed)
    telemetry_summary = _build_telemetry_model_summary(telemetry_df)
    telemetry_benchmarks = _build_telemetry_benchmark_table(telemetry_df)

    _save_table(benchmark_df, output_dir / "benchmark_scores.csv")
    _save_table(model_summary, output_dir / "model_summary.csv")
    _save_table(family_summary, output_dir / "family_summary.csv")
    _save_table(ranking_table, output_dir / "benchmark_rankings.csv")
    _save_table(pairwise, output_dir / "pairwise_wins.csv")
    _save_table(bootstrap_df, output_dir / "bootstrap_summary.csv")
    _save_table(telemetry_summary, output_dir / "telemetry_model_summary.csv")
    _save_table(telemetry_benchmarks, output_dir / "telemetry_by_benchmark.csv")

    _plot_model_mean_scores(model_summary, output_dir)
    _plot_model_normalized_scores(model_summary, output_dir)
    _plot_family_grouped_bars(family_summary, output_dir)
    _plot_score_heatmap(benchmark_df, output_dir, normalized=False)
    _plot_score_heatmap(benchmark_df, output_dir, normalized=True)
    _plot_rank_heatmap(benchmark_df, output_dir)
    _plot_pairwise_heatmap(pairwise, output_dir)
    _plot_distribution_boxplot(benchmark_df, output_dir)
    _plot_benchmark_spread(benchmark_df, output_dir)
    _plot_first_place_finishes(model_summary, output_dir)
    _plot_bootstrap_intervals(bootstrap_df, output_dir)
    _plot_radar(family_summary, output_dir)
    _plot_family_small_multiples(benchmark_df, output_dir)
    _plot_telemetry_bar(
        telemetry_summary,
        output_dir,
        value_col="mean_wall_clock_seconds",
        filename="telemetry_mean_wall_clock_seconds.png",
        title="Mean Wall-Clock Time Per Benchmark",
        ylabel="Seconds",
    )
    _plot_telemetry_bar(
        telemetry_summary,
        output_dir,
        value_col="mean_generation_seconds",
        filename="telemetry_mean_generation_seconds.png",
        title="Mean Generation Time Per Benchmark",
        ylabel="Seconds",
    )
    _plot_telemetry_bar(
        telemetry_summary,
        output_dir,
        value_col="mean_tokens_per_second",
        filename="telemetry_mean_tokens_per_second.png",
        title="Mean Tokens Per Second",
        ylabel="Tokens / Second",
    )
    _plot_telemetry_bar(
        telemetry_summary,
        output_dir,
        value_col="peak_cpu_ram_bytes",
        filename="telemetry_peak_cpu_ram_gib.png",
        title="Peak CPU RAM by Model",
        ylabel="Bytes",
    )
    _plot_telemetry_by_benchmark(
        telemetry_benchmarks,
        output_dir,
        value_col="wall_clock_time_per_sample_seconds_mean",
        filename="telemetry_wall_clock_by_benchmark.png",
        title="Wall-Clock Time by Benchmark",
        ylabel="Seconds",
    )
    _plot_telemetry_by_benchmark(
        telemetry_benchmarks,
        output_dir,
        value_col="peak_cpu_ram_bytes",
        filename="telemetry_peak_cpu_ram_by_benchmark.png",
        title="Peak CPU RAM by Benchmark",
        ylabel="Bytes",
    )
    _plot_stats_coverage(telemetry_summary, output_dir)
    _write_markdown_summary(
        benchmark_df,
        model_summary,
        family_summary,
        pairwise,
        bootstrap_df,
        telemetry_summary,
        output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a richer cross-model comparison report from saved results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="comparison/output")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    build_report(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
