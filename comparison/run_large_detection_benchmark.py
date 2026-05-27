from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from benchmarks import OpenImagesV4DetectionBenchmark
from models import GPT4, LlavaOnevision, Qwen25VL72B


OUTPUT_DIR = Path("comparison/output")
RESULTS_DIR = OUTPUT_DIR / "large_detection_results"
BENCHMARK_NAME = "openimages_v4_detection"


def _write_payload(model_name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}_{BENCHMARK_NAME}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run_one(model_name: str, model_factory, max_new_tokens: int) -> dict:
    benchmark = OpenImagesV4DetectionBenchmark(streaming=True)
    model = model_factory(max_new_tokens)
    report = benchmark.run(
        model=model,
        n=1,
        label_sample_size=4,
        show_progress=False,
    )
    payload = {
        "model": model_name,
        "benchmark": BENCHMARK_NAME,
        "report": report,
    }
    path = _write_payload(model_name, payload)
    result = report["results"][0]
    return {
        "model": model_name,
        "status": "ok",
        "results_path": str(path),
        "prediction": result.get("prediction", ""),
        "correct": bool(result.get("correct")),
        "f1": float(result.get("f1", 0.0) or 0.0),
    }


def _make_gpt41(max_new_tokens: int):
    return GPT4(model_id="gpt-4.1", max_new_tokens=max_new_tokens, temperature=0.0)


def _make_qwen72(max_new_tokens: int):
    return Qwen25VL72B(max_new_tokens=max_new_tokens)


def _make_llava72(max_new_tokens: int):
    return LlavaOnevision(
        model_id="llava-hf/llava-onevision-qwen2-72b-ov-hf",
        max_new_tokens=max_new_tokens,
        stream=False,
    )


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the GPT-4.1 run.")

    summary = []
    runs = [
        ("gpt-4.1", _make_gpt41, 128),
        ("llava-onevision-qwen2-72b-ov-hf", _make_llava72, 64),
        ("qwen2.5-vl-72b-instruct", _make_qwen72, 64),
    ]
    for model_name, factory, max_new_tokens in runs:
        try:
            print(f"=== {model_name} ===", flush=True)
            summary.append(_run_one(model_name, factory, max_new_tokens))
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
    summary_path = OUTPUT_DIR / "large_detection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
