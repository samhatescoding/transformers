from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from models import (
    Llama3LlavaNext8B,
    Llava,
    Llava15_13B,
    LlavaNextMistral7B,
    LlavaNextVicuna13B,
    LlavaOnevision,
    LlavaOnevision15_4BInstruct,
    LlavaOnevisionQwen2_7B,
    SmallLlava,
)
from runners.full_suite import run_full_suite

RESULTS_DIR = Path("results")
SUMMARY_PATH = Path(".tmp") / "llava_benchmark_summary.json"


def _llava_15_7b():
    return Llava(stream=False)


def _llava_15_13b():
    return Llava15_13B(stream=False)


def _llava_gemma_2b():
    return SmallLlava(stream=False)


def _llava_next_mistral_7b():
    return LlavaNextMistral7B()


def _llava_next_vicuna_13b():
    return LlavaNextVicuna13B()


def _llama3_llava_next_8b():
    return Llama3LlavaNext8B()


def _llava_onevision_qwen2_7b():
    return LlavaOnevisionQwen2_7B(stream=False)


def _llava_onevision_qwen2_72b():
    return LlavaOnevision(stream=False)


def _llava_onevision15_4b():
    return LlavaOnevision15_4BInstruct()


MODEL_FACTORIES = {
    "llava-gemma-2b": _llava_gemma_2b,
    "llava-1.5-7b-hf": _llava_15_7b,
    "llava-1.5-13b-hf": _llava_15_13b,
    "llava-v1.6-mistral-7b-hf": _llava_next_mistral_7b,
    "llava-v1.6-vicuna-13b-hf": _llava_next_vicuna_13b,
    "llama3-llava-next-8b-hf": _llama3_llava_next_8b,
    "llava-onevision-qwen2-7b-ov-hf": _llava_onevision_qwen2_7b,
    "llava-onevision-1.5-4b-instruct": _llava_onevision15_4b,
    "llava-onevision-qwen2-72b-ov-hf": _llava_onevision_qwen2_72b,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full benchmark suite for LLaVA-family vision models.")
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_FACTORIES), default=list(MODEL_FACTORIES))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_factories = {name: MODEL_FACTORIES[name] for name in args.models}
    summary = run_full_suite(
        model_factories=selected_factories,
        output_dir=RESULTS_DIR,
        summary_path=SUMMARY_PATH,
        num_samples=args.num_samples,
        overwrite=args.overwrite,
        streaming=not args.no_streaming,
    )
    counts = Counter(item["status"] for item in summary)
    print(f"Completed: {dict(counts)}. Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
