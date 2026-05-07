from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmark_runner import BENCHMARK_TOKENS, run_benchmark_matrix


def main() -> int:
    root = Path(__file__).resolve().parent
    model_name = sys.argv[1] if len(sys.argv) > 1 else "small-llava"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    for benchmark_name, max_new_tokens in BENCHMARK_TOKENS.items():
        result_path = root / "results" / f"{model_name}_{benchmark_name}.json"
        if result_path.exists():
            print(f"=== {benchmark_name} ({max_new_tokens}) ===", flush=True)
            print(f"[SKIP] Existing result: {result_path}", flush=True)
            continue
        print(f"=== {benchmark_name} ({max_new_tokens}) ===", flush=True)
        try:
            summaries = run_benchmark_matrix(
                model_names=[model_name],
                benchmark_names=[benchmark_name],
                num_samples=num_samples,
                benchmark_tokens={**BENCHMARK_TOKENS, benchmark_name: max_new_tokens},
                output_dir=root / "results",
            )
            result_path = Path(str(summaries[0]["results_path"]))
        except Exception as exc:
            print(f"[ERROR] {benchmark_name} failed: {exc}", flush=True)
            continue

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            report = payload["report"]
            first_result = report["results"][0]
            print(f"[PREDICTION] {first_result['prediction']}", flush=True)
            print(f"[STATS] {json.dumps(report.get('stats', {}), ensure_ascii=False)}", flush=True)
        except Exception as exc:
            print(f"[ERROR] Could not read result for {benchmark_name}: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
