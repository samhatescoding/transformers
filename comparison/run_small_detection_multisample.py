from __future__ import annotations

import json
import traceback
from pathlib import Path

from benchmarks import OpenImagesV4DetectionBenchmark
from models import Gemma, Qwen25VL3B, SmallLlava


OUTPUT_DIR = Path("comparison/output")
RESULTS_DIR = OUTPUT_DIR / "small_detection_multisample"
BENCHMARK_NAME = "openimages_v4_detection"
NUM_SAMPLES = 5
LABEL_SAMPLE_SIZE = 4


def _write_payload(model_name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}_{BENCHMARK_NAME}_n{NUM_SAMPLES}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _summarize_report(model_name: str, report: dict, path: Path) -> dict:
    results = report.get("results", [])
    if not results:
        return {
            "model": model_name,
            "status": "ok",
            "results_path": str(path),
            "num_samples": 0,
            "accuracy": 0.0,
            "mean_f1": 0.0,
            "mean_iou_matched": 0.0,
            "mean_iou_all_predictions": 0.0,
        }

    def _mean(key: str) -> float:
        values = [float(item.get(key, 0.0) or 0.0) for item in results]
        return sum(values) / len(values)

    accuracy = sum(1.0 if item.get("correct") else 0.0 for item in results) / len(results)
    return {
        "model": model_name,
        "status": "ok",
        "results_path": str(path),
        "num_samples": len(results),
        "accuracy": accuracy,
        "mean_f1": _mean("f1"),
        "mean_iou_matched": _mean("mean_iou_matched"),
        "mean_iou_all_predictions": _mean("mean_iou_all_predictions"),
    }


def main() -> int:
    benchmark = OpenImagesV4DetectionBenchmark(streaming=True)
    model_factories = {
        "llava-gemma-2b": lambda: SmallLlava(max_new_tokens=32, stream=False),
        "paligemma-3b-mix-224": lambda: Gemma(max_new_tokens=32),
        "qwen2.5-vl-3b-instruct": lambda: Qwen25VL3B(max_new_tokens=32),
    }

    summary = []
    for model_name, factory in model_factories.items():
        try:
            print(f"=== {model_name} ===", flush=True)
            model = factory()
            report = benchmark.run(
                model=model,
                n=NUM_SAMPLES,
                label_sample_size=LABEL_SAMPLE_SIZE,
                show_progress=False,
            )
            payload = {
                "model": model_name,
                "benchmark": BENCHMARK_NAME,
                "num_samples_requested": NUM_SAMPLES,
                "report": report,
            }
            path = _write_payload(model_name, payload)
            summary.append(_summarize_report(model_name, report, path))
        except Exception as exc:
            summary.append(
                {
                    "model": model_name,
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] {model_name}: {exc.__class__.__name__}: {exc}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "small_detection_multisample_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
