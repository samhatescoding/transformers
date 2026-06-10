from __future__ import annotations

import gc
import json
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import torch

from benchmarks import (
    BLIP3o60kBenchmark,
    ConceptualCaptionsBenchmark,
    ConceptualCaptionsCaptionBenchmark,
    CityscapesBenchmark,
    DFDCBenchmark,
    DiffusionDBBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    Flickr30kBenchmark,
    Flickr30kEntitiesBenchmark,
    FlyingThings3DBenchmark,
    GQABenchmark,
    HDTFBenchmark,
    HQEditBenchmark,
    ImageNet1kBenchmark,
    ImgEditBenchmark,
    INaturalistBenchmark,
    INaturalistDetectionBenchmark,
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    LVISBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    MagicBrushBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    OpenVid1MBenchmark,
    OpenVid1MCaptionBenchmark,
    PickAPicBenchmark,
    PlacesBenchmark,
    ShareGPT4oImageBenchmark,
    ShareGPT4oImageEditBenchmark,
    TAD66KBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VisualCoTBenchmark,
    VisualCoTDetectionBenchmark,
    VisualGenomeBenchmark,
    VQAv2Benchmark,
)

FULL_BENCHMARK_CLASSES = [
    BLIP3o60kBenchmark,
    ConceptualCaptionsBenchmark,
    ConceptualCaptionsCaptionBenchmark,
    CityscapesBenchmark,
    DFDCBenchmark,
    DiffusionDBBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    Flickr30kBenchmark,
    Flickr30kEntitiesBenchmark,
    FlyingThings3DBenchmark,
    GQABenchmark,
    HDTFBenchmark,
    HQEditBenchmark,
    ImageNet1kBenchmark,
    ImgEditBenchmark,
    INaturalistBenchmark,
    INaturalistDetectionBenchmark,
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    LVISBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    MagicBrushBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    OpenVid1MBenchmark,
    OpenVid1MCaptionBenchmark,
    PickAPicBenchmark,
    PlacesBenchmark,
    ShareGPT4oImageBenchmark,
    ShareGPT4oImageEditBenchmark,
    TAD66KBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VisualCoTBenchmark,
    VisualCoTDetectionBenchmark,
    VisualGenomeBenchmark,
    VQAv2Benchmark,
]


def _write_result(output_dir: Path, model_name: str, benchmark_name: str, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_{benchmark_name}.json"
    path.write_text(
        json.dumps({"model": model_name, "benchmark": benchmark_name, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_summary(summary_path: Path, summary: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_full_suite(
    *,
    model_factories: Mapping[str, Callable[[], object]],
    output_dir: Path,
    summary_path: Path,
    num_samples: int = 10,
    overwrite: bool = False,
    streaming: bool = True,
) -> list[dict]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    run_summary: list[dict] = []
    for model_name, factory in model_factories.items():
        try:
            model = factory()
        except Exception as exc:
            for benchmark_cls in FULL_BENCHMARK_CLASSES:
                benchmark_name = str(benchmark_cls.benchmark_name)
                result_path = output_dir / f"{model_name}_{benchmark_name}.json"
                if result_path.exists() and not overwrite:
                    continue
                run_summary.append(
                    {
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "status": "model_load_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            _write_summary(summary_path, run_summary)
            print(f"[MODEL_LOAD_ERROR] {model_name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        try:
            for benchmark_cls in FULL_BENCHMARK_CLASSES:
                benchmark_name = str(benchmark_cls.benchmark_name)
                result_path = output_dir / f"{model_name}_{benchmark_name}.json"
                if result_path.exists() and not overwrite:
                    run_summary.append(
                        {"model": model_name, "benchmark": benchmark_name, "status": "skipped", "path": str(result_path)}
                    )
                    continue

                print(f"[RUN] {model_name} / {benchmark_name}", flush=True)
                started_at = perf_counter()
                try:
                    benchmark = benchmark_cls(streaming=streaming)
                    model.max_new_tokens = benchmark.default_max_new_tokens
                    sample_count = min(num_samples, 9) if benchmark_name == "ucf101" else num_samples
                    report = benchmark.run(
                        model=model,
                        n=sample_count,
                        label_sample_size=max(4, sample_count),
                        show_progress=False,
                    )
                    path = _write_result(output_dir, model_name, benchmark_name, report)
                    entry = {
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "status": "ok",
                        "elapsed_seconds": perf_counter() - started_at,
                        "path": str(path),
                    }
                except Exception as exc:
                    entry = {
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "status": "error",
                        "elapsed_seconds": perf_counter() - started_at,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                run_summary.append(entry)
                _write_summary(summary_path, run_summary)
                print(f"[{entry['status'].upper()}] {model_name} / {benchmark_name}", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_summary(summary_path, run_summary)
    return run_summary
