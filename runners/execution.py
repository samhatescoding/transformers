from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .benchmark_run import BenchmarkRun
from .model_run import ModelRun


def _write_report(output_dir: Path, model_name: str, benchmark_name: str, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_{benchmark_name}.json"
    path.write_text(
        json.dumps({"model": model_name, "benchmark": benchmark_name, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_benchmark_matrix(
    models: Sequence[ModelRun],
    benchmark_runs: Sequence[BenchmarkRun],
    *,
    output_dir: str | Path = "results",
) -> list[dict[str, object]]:
    resolved_output_dir = Path(output_dir)
    summaries: list[dict[str, object]] = []

    for model_run in models:
        model = model_run.model
        model_name = model_run.name
        for benchmark_run in benchmark_runs:
            benchmark = benchmark_run.benchmark
            report = benchmark.run(
                model=model,
                n=benchmark_run.num_samples,
                label_sample_size=max(4, benchmark_run.num_samples),
                show_progress=False,
            )
            report.setdefault("stats", {})
            if model_run.load_time_seconds is not None:
                report["stats"]["model_load_time_seconds"] = model_run.load_time_seconds
            result_path = _write_report(
                output_dir=resolved_output_dir,
                model_name=model_name,
                benchmark_name=benchmark.name,
                report=report,
            )
            summaries.append(
                {
                    "model": model_name,
                    "benchmark": benchmark.name,
                    "num_samples": benchmark_run.num_samples,
                    "max_new_tokens": getattr(model, "max_new_tokens", None),
                    "results_path": str(result_path),
                }
            )

    return summaries


def run_benchmark_runs(
    models: Sequence[ModelRun],
    benchmark_runs: Sequence[BenchmarkRun],
    *,
    output_dir: str | Path = "results",
) -> list[dict[str, object]]:
    return run_benchmark_matrix(models=models, benchmark_runs=benchmark_runs, output_dir=output_dir)
