from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from models import GPT4, GPT5, GPT51, GPT52, GPT53ChatLatest, GPT54, GPT54Mini, GPT54Nano, GPT55
from runners.full_suite import run_full_suite

RESULTS_DIR = Path("results")
SUMMARY_PATH = Path(".tmp") / "untested_gpt_benchmark_summary.json"

MODEL_CLASSES = {
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all concrete benchmarks for GPT models without prior results.")
    parser.add_argument("--models", nargs="*", choices=sorted(MODEL_CLASSES), default=list(MODEL_CLASSES))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    selected_factories = {
        name: (lambda cls=MODEL_CLASSES[name]: cls(max_new_tokens=16))
        for name in args.models
    }
    summary = run_full_suite(
        model_factories=selected_factories,
        output_dir=RESULTS_DIR,
        summary_path=SUMMARY_PATH,
        num_samples=args.num_samples,
        overwrite=args.overwrite,
        streaming=not args.no_streaming,
    )
    print(f"Completed: {dict(Counter(item['status'] for item in summary))}. Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
