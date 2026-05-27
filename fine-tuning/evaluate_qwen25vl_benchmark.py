"""Evaluate base or LoRA-adapted Qwen2.5-VL-3B on any repository benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import Qwen25VL3B
from runners import BenchmarkRun, ModelRun, run_benchmark_matrix

from benchmark_registry import BENCHMARK_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_CLASSES))
    parser.add_argument("--split", help="Evaluation split; keep this separate from training data.")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--non-streaming", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/fine-tuning")
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1.")
    if args.adapter_path is not None and not args.adapter_path.exists():
        raise SystemExit(f"Adapter path does not exist: {args.adapter_path}")

    name = args.model_name or (
        args.adapter_path.name
        if args.adapter_path is not None
        else f"qwen2.5-vl-3b-base-{args.benchmark}"
    )
    benchmark_cls = BENCHMARK_CLASSES[args.benchmark]
    benchmark_kwargs = {"streaming": not args.non_streaming}
    if args.split is not None:
        benchmark_kwargs["split"] = args.split
    benchmark = benchmark_cls(**benchmark_kwargs)
    model = ModelRun.from_factory(
        name,
        Qwen25VL3B,
        max_new_tokens=benchmark.default_max_new_tokens,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    summaries = run_benchmark_matrix(
        models=[model],
        benchmark_runs=[
            BenchmarkRun(benchmark=benchmark, num_samples=args.samples)
        ],
        output_dir=args.output_dir,
    )
    print(summaries[0]["results_path"])


if __name__ == "__main__":
    main()

