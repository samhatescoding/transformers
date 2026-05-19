from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from benchmark_run import BenchmarkRun
from benchmark_runner import available_model_names, run_benchmark_matrix, run_benchmark_runs

_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "test_benchmark_runner"


class _StubModel:
    def __init__(self, name: str, max_new_tokens: int) -> None:
        self.name = name
        self.max_new_tokens = max_new_tokens


class _StubBenchmark:
    def __init__(self, name: str, seen_tokens: list[int], streaming: bool = True) -> None:
        self.name = name
        self._seen_tokens = seen_tokens
        self._streaming = streaming

    def run(self, model, n: int, label_sample_size: int, show_progress: bool) -> dict:
        self._seen_tokens.append(model.max_new_tokens)
        return {
            "benchmark": self.name,
            "dataset": f"{self.name}_dataset",
            "num_samples": n,
            "num_candidate_labels": label_sample_size,
            "results": [
                {
                    "index": 1,
                    "prediction": f"{model.name}:{self.name}",
                    "correct": True,
                    "valid_labels": ["ok"],
                }
            ],
            "stats": {"streaming": self._streaming},
        }


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
        _TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)

    def test_run_benchmark_matrix_runs_every_model_against_every_benchmark(self) -> None:
        bench_a_tokens: list[int] = []
        bench_b_tokens: list[int] = []

        benchmark_registry = {
            "bench_a": lambda streaming=True: _StubBenchmark("bench_a", bench_a_tokens, streaming=streaming),
            "bench_b": lambda streaming=True: _StubBenchmark("bench_b", bench_b_tokens, streaming=streaming),
        }
        model_registry = {
            "model_a": lambda max_new_tokens: _StubModel("model_a", max_new_tokens),
            "model_b": lambda max_new_tokens: _StubModel("model_b", max_new_tokens),
        }

        output_dir = _TMP_ROOT / "matrix"
        summaries = run_benchmark_matrix(
            model_names=["model_a", "model_b"],
            benchmark_names=["bench_a", "bench_b"],
            num_samples=3,
            benchmark_registry=benchmark_registry,
            model_registry=model_registry,
            benchmark_tokens={"bench_a": 11, "bench_b": 17},
            output_dir=output_dir,
        )

        self.assertEqual(len(summaries), 4)
        self.assertEqual(bench_a_tokens, [11, 11])
        self.assertEqual(bench_b_tokens, [17, 17])

        result_paths = sorted(Path(item["results_path"]) for item in summaries)
        self.assertEqual(len(result_paths), 4)
        for path in result_paths:
            self.assertTrue(path.exists(), str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["report"]["num_samples"], 3)
            self.assertIn(payload["benchmark"], {"bench_a", "bench_b"})
            self.assertIn(payload["model"], {"model_a", "model_b"})
            self.assertIn("model_load_time_seconds", payload["report"]["stats"])

    def test_run_benchmark_matrix_rejects_unknown_names(self) -> None:
        output_dir = _TMP_ROOT / "errors"

        with self.assertRaisesRegex(ValueError, "Unknown models"):
            run_benchmark_matrix(
                model_names=["missing_model"],
                benchmark_names=[],
                num_samples=1,
                benchmark_registry={},
                model_registry={},
                output_dir=output_dir,
            )

        with self.assertRaisesRegex(ValueError, "Unknown benchmarks"):
            run_benchmark_matrix(
                model_names=[],
                benchmark_names=["missing_benchmark"],
                num_samples=1,
                benchmark_registry={},
                model_registry={},
                output_dir=output_dir,
            )

    def test_benchmark_run_rejects_invalid_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_samples must be at least 1"):
            BenchmarkRun(benchmark=_StubBenchmark("bench_a", []), num_samples=0)

    def test_run_benchmark_runs_executes_prepared_objects(self) -> None:
        bench_a_tokens: list[int] = []
        bench_b_tokens: list[int] = []
        output_dir = _TMP_ROOT / "object_runs"

        model = _StubModel("prepared_model", 23)
        benchmark_runs = [
            BenchmarkRun(benchmark=_StubBenchmark("bench_a", bench_a_tokens), num_samples=2),
            BenchmarkRun(benchmark=_StubBenchmark("bench_b", bench_b_tokens), num_samples=5),
        ]

        summaries = run_benchmark_runs(
            models=[model],
            benchmark_runs=benchmark_runs,
            output_dir=output_dir,
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(bench_a_tokens, [23])
        self.assertEqual(bench_b_tokens, [23])
        self.assertEqual(summaries[0]["num_samples"], 2)
        self.assertEqual(summaries[1]["num_samples"], 5)
        self.assertEqual(summaries[0]["max_new_tokens"], 23)
        self.assertEqual(summaries[1]["max_new_tokens"], 23)

        for item in summaries:
            path = Path(str(item["results_path"]))
            self.assertTrue(path.exists(), str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "prepared_model")
            self.assertIn(payload["benchmark"], {"bench_a", "bench_b"})

    def test_available_model_names_exposes_large_vlm_variants(self) -> None:
        names = available_model_names()
        self.assertIn("qwen25-vl-7b", names)
        self.assertIn("qwen25-vl-72b", names)
        self.assertIn("llava-onevision-72b", names)
        self.assertIn("internvl25-8b", names)
        self.assertIn("minicpm-v-2_6", names)
