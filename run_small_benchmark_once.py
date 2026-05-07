import json
import sys
from pathlib import Path

from benchmark_runner import (
    BENCHMARK_TOKENS,
    available_benchmark_names,
    available_model_names,
    run_benchmark_matrix,
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python run_small_benchmark_once.py <benchmark_name> [max_new_tokens] [model_name] [num_samples]")
        return 2

    benchmark_name = argv[1]
    max_new_tokens = int(argv[2]) if len(argv) > 2 else 24
    model_name = argv[3] if len(argv) > 3 else "small-llava"
    num_samples = int(argv[4]) if len(argv) > 4 else 1
    benchmark_names = set(available_benchmark_names())
    model_names = set(available_model_names())
    if benchmark_name not in benchmark_names:
        print(f"Unknown benchmark: {benchmark_name}")
        return 2
    if model_name not in model_names:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(sorted(model_names))}")
        return 2

    summaries = run_benchmark_matrix(
        model_names=[model_name],
        benchmark_names=[benchmark_name],
        num_samples=num_samples,
        benchmark_tokens={**BENCHMARK_TOKENS, benchmark_name: max_new_tokens},
    )
    path = summaries[0]["results_path"]

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = payload["report"]["results"][0]
    print(path)
    print(result.get("prediction", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
