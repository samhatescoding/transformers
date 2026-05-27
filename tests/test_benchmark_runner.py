from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import models
from runners.benchmark_run import BenchmarkRun
from runners.execution import run_benchmark_matrix, run_benchmark_runs
from runners.model_run import ModelRun
from models.falcon_11b_vlm import Falcon
from models.paligemma_3b_mix_224 import Gemma
from models.internvl2_5_8b import InternVL25
from models.llava15_7b import Llava
from models.llava_onevision_qwen2_72b_ov import LlavaOnevision
from models.llava_gemma_2b import SmallLlava
from models.minicpm_v_2_6 import MiniCPMV26
from models._qwen2_5_vl import _Qwen25VLBase
from models.qwen2_5_vl_3b_instruct import Qwen25VL3B
from models.qwen2_5_vl_72b_instruct import Qwen25VL72B

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
        output_dir = _TMP_ROOT / "matrix"

        models = [
            ModelRun(name="model_a", model=_StubModel("model_a", 11), load_time_seconds=0.1),
            ModelRun(name="model_b", model=_StubModel("model_b", 17), load_time_seconds=0.2),
        ]
        benchmark_runs = [
            BenchmarkRun(benchmark=_StubBenchmark("bench_a", bench_a_tokens), num_samples=3),
            BenchmarkRun(benchmark=_StubBenchmark("bench_b", bench_b_tokens), num_samples=3),
        ]

        summaries = run_benchmark_matrix(
            models=models,
            benchmark_runs=benchmark_runs,
            output_dir=output_dir,
        )

        self.assertEqual(len(summaries), 4)
        self.assertEqual(bench_a_tokens, [11, 17])
        self.assertEqual(bench_b_tokens, [11, 17])

        result_paths = sorted(Path(item["results_path"]) for item in summaries)
        self.assertEqual(len(result_paths), 4)
        for path in result_paths:
            self.assertTrue(path.exists(), str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["report"]["num_samples"], 3)
            self.assertIn(payload["benchmark"], {"bench_a", "bench_b"})
            self.assertIn(payload["model"], {"model_a", "model_b"})
            self.assertIn("model_load_time_seconds", payload["report"]["stats"])

    def test_benchmark_run_rejects_invalid_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_samples must be at least 1"):
            BenchmarkRun(benchmark=_StubBenchmark("bench_a", []), num_samples=0)

    def test_model_run_from_factory_measures_loading_time(self) -> None:
        model_run = ModelRun.from_factory("factory_model", _StubModel, "factory_model", 13)
        self.assertEqual(model_run.name, "factory_model")
        self.assertIsInstance(model_run.model, _StubModel)
        self.assertGreaterEqual(float(model_run.load_time_seconds or 0.0), 0.0)

    def test_run_benchmark_runs_executes_prepared_objects(self) -> None:
        bench_a_tokens: list[int] = []
        bench_b_tokens: list[int] = []
        output_dir = _TMP_ROOT / "object_runs"

        model = ModelRun(name="prepared_model", model=_StubModel("prepared_model", 23), load_time_seconds=0.3)
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
            self.assertEqual(payload["report"]["stats"]["model_load_time_seconds"], 0.3)

    def test_qwen_checkpoint_subclasses_use_explicit_model_ids(self) -> None:
        def fake_init(self, max_new_tokens=100, temperature=0.0):
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature

        with patch.object(_Qwen25VLBase, "__init__", fake_init):
            qwen3b = Qwen25VL3B(max_new_tokens=16)
            qwen72b = Qwen25VL72B(max_new_tokens=32)

        self.assertEqual(qwen3b.model_id, "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(qwen3b.name, "qwen2.5-vl-3b-instruct")
        self.assertEqual(qwen3b.max_new_tokens, 16)
        self.assertEqual(qwen72b.model_id, "Qwen/Qwen2.5-VL-72B-Instruct")
        self.assertEqual(qwen72b.name, "qwen2.5-vl-72b-instruct")
        self.assertEqual(qwen72b.max_new_tokens, 32)

    def test_qwen_checkpoint_can_record_lora_adapter_path(self) -> None:
        with patch.object(_Qwen25VLBase, "_load_input_artifact_and_model"):
            model = Qwen25VL3B(adapter_path="fine-tuning/output/adapter")
        self.assertEqual(model.adapter_path, "fine-tuning/output/adapter")

    def test_default_checkpoint_names_are_specific(self) -> None:
        self.assertEqual(Llava._name_from_model_id("llava-hf/llava-1.5-7b-hf"), "llava-1.5-7b-hf")
        self.assertEqual(Llava._name_from_model_id("Intel/llava-gemma-2b"), "llava-gemma-2b")
        self.assertEqual(SmallLlava.default_model_id, "Intel/llava-gemma-2b")
        self.assertEqual(
            LlavaOnevision._name_from_model_id("llava-hf/llava-onevision-qwen2-72b-ov-hf"),
            "llava-onevision-qwen2-72b-ov-hf",
        )
        self.assertEqual(Gemma._name_from_model_id("google/paligemma-3b-mix-224"), "paligemma-3b-mix-224")
        self.assertEqual(Falcon._name_from_model_id("tiiuae/falcon-11B-vlm"), "falcon-11b-vlm")
        self.assertEqual(InternVL25._name_from_model_id("OpenGVLab/InternVL2_5-8B"), "internvl2.5-8b")
        self.assertEqual(MiniCPMV26._name_from_model_id("openbmb/MiniCPM-V-2_6"), "minicpm-v-2.6")

    def test_all_catalog_model_wrappers_are_public_and_identify_their_checkpoint(self) -> None:
        expected_defaults = {
            "GPT41": "gpt-4.1",
            "GPT5": "gpt-5",
            "GPT51": "gpt-5.1",
            "GPT52": "gpt-5.2",
            "GPT53ChatLatest": "gpt-5.3-chat-latest",
            "GPT54": "gpt-5.4",
            "GPT54Mini": "gpt-5.4-mini",
            "GPT54Nano": "gpt-5.4-nano",
            "GPT55": "gpt-5.5",
            "O3": "o3",
            "Llava15_13B": "llava-hf/llava-1.5-13b-hf",
            "LlavaNextMistral7B": "llava-hf/llava-v1.6-mistral-7b-hf",
            "LlavaNextVicuna13B": "llava-hf/llava-v1.6-vicuna-13b-hf",
            "Llama3LlavaNext8B": "llava-hf/llama3-llava-next-8b-hf",
            "LlavaOnevisionQwen2_7B": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "LlavaOnevision15_4BInstruct": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
            "Qwen25VL7B": "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen25VL32B": "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen3VL4B": "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen3VL8B": "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen35_4B": "Qwen/Qwen3.5-4B",
            "Qwen35_9B": "Qwen/Qwen3.5-9B",
            "PaliGemma3BMix448": "google/paligemma-3b-mix-448",
            "PaliGemma2_3BMix448": "google/paligemma2-3b-mix-448",
            "PaliGemma2_10BMix448": "google/paligemma2-10b-mix-448",
            "Gemma3_4B": "google/gemma-3-4b-it",
            "Gemma3_12B": "google/gemma-3-12b-it",
            "Gemma3_27B": "google/gemma-3-27b-it",
            "Gemma4E2B": "google/gemma-4-E2B-it",
            "Gemma4E4B": "google/gemma-4-E4B-it",
            "Gemma4_26BA4B": "google/gemma-4-26B-A4B-it",
            "Gemma4_31B": "google/gemma-4-31B-it",
            "InternVL25_4B": "OpenGVLab/InternVL2_5-4B",
            "InternVL3_2B": "OpenGVLab/InternVL3-2B",
            "InternVL3_8B": "OpenGVLab/InternVL3-8B",
            "InternVL35_8BInstruct": "OpenGVLab/InternVL3_5-8B-Instruct",
            "MiniCPMo26": "openbmb/MiniCPM-o-2_6",
            "MiniCPMV46": "openbmb/MiniCPM-V-4.6",
            "MiniCPMV46Thinking": "openbmb/MiniCPM-V-4.6-Thinking",
        }
        for public_name, model_id in expected_defaults.items():
            cls = getattr(models, public_name)
            actual = getattr(cls, "default_model_id", getattr(cls, "model_id", None))
            self.assertEqual(actual, model_id, public_name)

    def test_gpt5_request_does_not_send_temperature(self) -> None:
        model = models.GPT55.__new__(models.GPT55)
        model.model_id = "gpt-5.5"
        model.max_new_tokens = 12
        model.temperature = 0.0
        requests = []
        model.client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: requests.append(kwargs) or SimpleNamespace(output_text="answer")
            )
        )

        answer = model.predict(Image.new("RGB", (2, 2), "white"), "Describe <image>")

        self.assertEqual(answer, "answer")
        self.assertNotIn("temperature", requests[0])
        self.assertEqual(requests[0]["reasoning"], {"effort": "none"})
        self.assertEqual(requests[0]["max_output_tokens"], 16)

    def test_gpt_models_can_raise_minimum_visible_answer_budget(self) -> None:
        self.assertEqual(models.GPT5.min_output_tokens, 32)
        self.assertEqual(models.GPT53ChatLatest.min_output_tokens, 64)
