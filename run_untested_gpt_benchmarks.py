from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter

from benchmarks import (
    BLIP3o60kBenchmark,
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
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    LVISBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    OpenVid1MBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VisualCoTBenchmark,
    VisualGenomeBenchmark,
    VQAv2Benchmark,
)
from models import GPT4, GPT5, GPT51, GPT52, GPT53ChatLatest, GPT54, GPT54Mini, GPT54Nano, GPT55

RESULTS_DIR = Path("results")
SUMMARY_PATH = Path(".tmp") / "untested_gpt_benchmark_summary.json"

MODEL_FACTORIES = {
    "gpt-4o": GPT4,
    "gpt-5": GPT5,
    "gpt-5.1": GPT51,
    "gpt-5.2": GPT52,
    "gpt-5.3-chat-latest": GPT53ChatLatest,
    "gpt-5.4": GPT54,
    "gpt-5.4-mini": GPT54Mini,
    "gpt-5.4-nano": GPT54Nano,
    "gpt-5.5": GPT55,
}

BENCHMARK_CLASSES = [
    BLIP3o60kBenchmark,
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
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    LVISBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    OpenVid1MBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VisualCoTBenchmark,
    VisualGenomeBenchmark,
    VQAv2Benchmark,
]


def _write_result(model_name: str, benchmark_name: str, report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{model_name}_{benchmark_name}.json"
    path.write_text(
        json.dumps({"model": model_name, "benchmark": benchmark_name, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_summary(summary: list[dict]) -> None:
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all concrete benchmarks for GPT models without prior results.")
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_FACTORIES), default=list(MODEL_FACTORIES))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    run_summary: list[dict] = []
    for model_name in args.models:
        model = MODEL_FACTORIES[model_name](max_new_tokens=16)
        for benchmark_cls in BENCHMARK_CLASSES:
            benchmark_name = str(benchmark_cls.benchmark_name)
            result_path = RESULTS_DIR / f"{model_name}_{benchmark_name}.json"
            if result_path.exists() and not args.overwrite:
                run_summary.append({"model": model_name, "benchmark": benchmark_name, "status": "skipped", "path": str(result_path)})
                continue

            print(f"[RUN] {model_name} / {benchmark_name}", flush=True)
            started_at = perf_counter()
            try:
                benchmark = benchmark_cls(streaming=not args.no_streaming)
                model.max_new_tokens = benchmark.default_max_new_tokens
                num_samples = min(args.num_samples, 9) if benchmark_name == "ucf101" else args.num_samples
                report = benchmark.run(
                    model=model,
                    n=num_samples,
                    label_sample_size=max(4, num_samples),
                    show_progress=False,
                )
                path = _write_result(model_name, benchmark_name, report)
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
            _write_summary(run_summary)
            print(f"[{entry['status'].upper()}] {model_name} / {benchmark_name}", flush=True)

    _write_summary(run_summary)
    succeeded = sum(entry["status"] == "ok" for entry in run_summary)
    failed = sum(entry["status"] == "error" for entry in run_summary)
    skipped = sum(entry["status"] == "skipped" for entry in run_summary)
    print(f"Completed: {succeeded} succeeded, {failed} failed, {skipped} skipped. Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
