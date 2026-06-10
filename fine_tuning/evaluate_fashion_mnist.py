"""Evaluate an OpenAI vision model on the held-out Fashion-MNIST test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import FashionMNISTBenchmark
from models import GPT4
from runners import BenchmarkRun, ModelRun, run_benchmark_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/fine-tuning")
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1.")

    benchmark = FashionMNISTBenchmark(split="test", streaming=True)
    model_run = ModelRun.from_factory(
        args.model_name,
        GPT4,
        model_id=args.model_id,
        max_new_tokens=16,
    )
    summaries = run_benchmark_matrix(
        models=[model_run],
        benchmark_runs=[BenchmarkRun(benchmark=benchmark, num_samples=args.samples)],
        output_dir=args.output_dir,
    )
    print(summaries[0]["results_path"])


if __name__ == "__main__":
    main()
