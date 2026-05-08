from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

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
from models import Gemma, InternVL25, LlavaOnevision, MiniCPMV26, Qwen25VL, SmallLlava

RESULTS_DIR = Path("results")
TEST_SIZE = 1
LABEL_SAMPLE_SIZE = 4

BENCHMARK_FACTORIES = [
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


MODELS = {
    "small-llava": lambda: SmallLlava(max_new_tokens=32, stream=False, load_in_4bit=False),
    "gemma": lambda: Gemma(max_new_tokens=32),
    "qwen25-vl": lambda: Qwen25VL(max_new_tokens=32),
    "qwen25-vl-7b": lambda: Qwen25VL(model_id="Qwen/Qwen2.5-VL-7B-Instruct", max_new_tokens=32),
    "qwen25-vl-72b": lambda: Qwen25VL(model_id="Qwen/Qwen2.5-VL-72B-Instruct", max_new_tokens=32),
    "llava-onevision-72b": lambda: LlavaOnevision(
        model_id="llava-hf/llava-onevision-qwen2-72b-ov-hf",
        max_new_tokens=32,
        stream=False,
        load_in_4bit=False,
    ),
    "internvl25-8b": lambda: InternVL25(
        model_id="OpenGVLab/InternVL2_5-8B",
        max_new_tokens=32,
    ),
    "minicpm-v-2_6": lambda: MiniCPMV26(
        model_id="openbmb/MiniCPM-V-2_6",
        max_new_tokens=32,
    ),
}


def _write_payload(model_name: str, benchmark_name: str, report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}_{benchmark_name}.json"
    payload = {
        "model": model_name,
        "benchmark": benchmark_name,
        "report": report,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    requested_model = argv[1] if len(argv) > 1 else "small-llava"
    model_factory = MODELS.get(requested_model)
    if model_factory is None:
        print(f"Unknown model: {requested_model}")
        print(f"Available models: {', '.join(sorted(MODELS))}")
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_started_at = time.perf_counter()
    model = model_factory()
    model_load_time_seconds = time.perf_counter() - model_started_at
    model_name = getattr(model, "name", model.__class__.__name__)
    summary: list[dict[str, str]] = []

    for benchmark_factory in BENCHMARK_FACTORIES:
        benchmark_name = benchmark_factory.__name__
        try:
            benchmark = benchmark_factory()
            report = benchmark.run(
                model=model,
                n=TEST_SIZE,
                label_sample_size=LABEL_SAMPLE_SIZE,
                show_progress=False,
            )
            report.setdefault("stats", {})
            report["stats"]["model_load_time_seconds"] = model_load_time_seconds
            output_path = _write_payload(model_name=model_name, benchmark_name=benchmark.name, report=report)
            summary.append(
                {
                    "benchmark": benchmark.name,
                    "status": "ok",
                    "results_path": str(output_path),
                }
            )
            print(f"[OK] {benchmark.name}")
        except Exception as exc:
            summary.append(
                {
                    "benchmark": benchmark_name,
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] {benchmark_name}: {exc.__class__.__name__}: {exc}")

    summary_path = RESULTS_DIR / f"{model_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
