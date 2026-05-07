from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from benchmarks._base_benchmark import BaseBenchmark
from benchmarks import (
    ConceptualCaptionsBenchmark,
    ConceptualCaptionsCaptionBenchmark,
    DFDCBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    Flickr30kBenchmark,
    Flickr30kEntitiesBenchmark,
    GQABenchmark,
    ImageNet1kBenchmark,
    INaturalistBenchmark,
    LSUNBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VQAv2Benchmark,
)
from models import Gemma, GPT4, Llava, Qwen25VL, SmallLlava

BenchmarkFactory = Callable[..., BaseBenchmark]
ModelFactory = Callable[[int], object]

@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: ModelFactory


BENCHMARKS: tuple[BenchmarkFactory, ...] = (
    ConceptualCaptionsBenchmark,
    ConceptualCaptionsCaptionBenchmark,
    DFDCBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    Flickr30kBenchmark,
    Flickr30kEntitiesBenchmark,
    GQABenchmark,
    ImageNet1kBenchmark,
    INaturalistBenchmark,
    LSUNBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VQAv2Benchmark,
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="small-llava",
        factory=lambda max_new_tokens: SmallLlava(
            max_new_tokens=max_new_tokens,
            stream=False,
            load_in_4bit=False,
        ),
    ),
    ModelSpec(
        name="llava-7b",
        factory=lambda max_new_tokens: Llava(
            model_id="llava-hf/llava-1.5-7b-hf",
            max_new_tokens=max_new_tokens,
            stream=False,
            load_in_4bit=False,
        ),
    ),
    ModelSpec(
        name="gemma",
        factory=lambda max_new_tokens: Gemma(
            max_new_tokens=max_new_tokens,
        ),
    ),
    ModelSpec(
        name="qwen25-vl",
        factory=lambda max_new_tokens: Qwen25VL(
            max_new_tokens=max_new_tokens,
        ),
    ),
    ModelSpec(
        name="gpt-4.1",
        factory=lambda max_new_tokens: GPT4(
            model_id="gpt-4.1",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        ),
    ),
)


def _benchmark_name(benchmark_factory: BenchmarkFactory) -> str:
    name = getattr(benchmark_factory, "benchmark_name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Benchmark factory {benchmark_factory!r} is missing a benchmark_name attribute")
    return name


def _benchmark_default_tokens(benchmark_factory: BenchmarkFactory) -> int:
    value = getattr(benchmark_factory, "default_max_new_tokens", 24)
    return int(value)


def _resolve_benchmark_registry(
    benchmark_registry: Mapping[str, BenchmarkFactory] | Sequence[BenchmarkFactory],
) -> dict[str, BenchmarkFactory]:
    if isinstance(benchmark_registry, Mapping):
        return dict(benchmark_registry)
    return {_benchmark_name(factory): factory for factory in benchmark_registry}


def _resolve_model_registry(
    model_registry: Mapping[str, ModelFactory] | Sequence[ModelSpec],
) -> dict[str, ModelFactory]:
    if isinstance(model_registry, Mapping):
        return dict(model_registry)
    return {spec.name: spec.factory for spec in model_registry}


def available_benchmark_names(
    benchmark_registry: Mapping[str, BenchmarkFactory] | Sequence[BenchmarkFactory] | None = None,
) -> list[str]:
    registry = _resolve_benchmark_registry(BENCHMARKS if benchmark_registry is None else benchmark_registry)
    return sorted(registry)


def available_model_names(
    model_registry: Mapping[str, ModelFactory] | Sequence[ModelSpec] | None = None,
) -> list[str]:
    registry = _resolve_model_registry(MODELS if model_registry is None else model_registry)
    return sorted(registry)


def get_benchmark_token_defaults(
    benchmark_registry: Mapping[str, BenchmarkFactory] | Sequence[BenchmarkFactory] | None = None,
) -> dict[str, int]:
    source = BENCHMARKS if benchmark_registry is None else benchmark_registry
    if isinstance(source, Mapping):
        return {name: 24 for name in source}
    return {_benchmark_name(factory): _benchmark_default_tokens(factory) for factory in source}


BENCHMARK_TOKENS = get_benchmark_token_defaults()


def _write_report(output_dir: Path, model_name: str, benchmark_name: str, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_{benchmark_name}.json"
    path.write_text(
        json.dumps({"model": model_name, "benchmark": benchmark_name, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_benchmark_matrix(
    model_names: list[str],
    benchmark_names: list[str],
    num_samples: int,
    *,
    benchmark_registry: Mapping[str, BenchmarkFactory] | Sequence[BenchmarkFactory] | None = None,
    model_registry: Mapping[str, ModelFactory] | Sequence[ModelSpec] | None = None,
    benchmark_tokens: dict[str, int] | None = None,
    output_dir: str | Path = "results",
) -> list[dict[str, object]]:
    benchmark_registry = _resolve_benchmark_registry(BENCHMARKS if benchmark_registry is None else benchmark_registry)
    model_registry = _resolve_model_registry(MODELS if model_registry is None else model_registry)
    if benchmark_tokens is None:
        benchmark_tokens = get_benchmark_token_defaults(benchmark_registry)
    resolved_output_dir = Path(output_dir)
    summaries: list[dict[str, object]] = []

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    unknown_models = [name for name in model_names if name not in model_registry]
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")

    unknown_benchmarks = [name for name in benchmark_names if name not in benchmark_registry]
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmarks: {', '.join(unknown_benchmarks)}")

    for model_name in model_names:
        initial_tokens = max((benchmark_tokens.get(name, 24) for name in benchmark_names), default=24)
        model_started_at = time.perf_counter()
        model = model_registry[model_name](initial_tokens)
        model_load_time_seconds = time.perf_counter() - model_started_at
        for benchmark_name in benchmark_names:
            benchmark_factory = benchmark_registry[benchmark_name]
            max_new_tokens = benchmark_tokens.get(benchmark_name, initial_tokens)
            if hasattr(model, "max_new_tokens"):
                model.max_new_tokens = max_new_tokens
            benchmark = benchmark_factory(streaming=True)
            report = benchmark.run(
                model=model,
                n=num_samples,
                label_sample_size=max(4, num_samples),
                show_progress=False,
            )
            report.setdefault("stats", {})
            report["stats"]["model_load_time_seconds"] = model_load_time_seconds
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
                    "num_samples": num_samples,
                    "max_new_tokens": max_new_tokens,
                    "results_path": str(result_path),
                }
            )

    return summaries
