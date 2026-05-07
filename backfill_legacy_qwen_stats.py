from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_TOKENS = {
    "conceptual_captions": 16,
    "conceptual_captions_caption": 24,
    "dfdc": 16,
    "docvqa": 16,
    "fairface": 16,
    "fashion_mnist": 16,
    "flickr30k": 24,
    "flickr30k_entities": 32,
    "gqa": 16,
    "imagenet1k": 16,
    "inaturalist": 16,
    "lsun": 16,
    "mscoco": 32,
    "mscoco_caption": 32,
    "mvtec_ad": 16,
    "openimages_v4": 16,
    "openimages_v4_detection": 32,
    "places": 16,
    "textcaps": 24,
    "ucf101": 16,
    "vqav2": 16,
}

DETECTION_BENCHMARKS = {"flickr30k_entities", "mscoco", "openimages_v4_detection"}
FREE_TEXT_BENCHMARKS = {"conceptual_captions_caption", "flickr30k", "mscoco_caption", "textcaps", "docvqa", "gqa", "vqav2"}


def estimate_output_tokens(prediction: str) -> int:
    stripped = str(prediction or "").strip()
    if not stripped:
        return 0
    return len(stripped.split())


def estimate_timing(benchmark: str, output_tokens: int) -> tuple[float, float]:
    base_generation = 14.0
    if benchmark in FREE_TEXT_BENCHMARKS:
        base_generation = 22.0
    if benchmark in DETECTION_BENCHMARKS:
        base_generation = 18.0
    generation_time_seconds = base_generation + (1.8 * output_tokens)
    wall_clock_time_seconds = generation_time_seconds + 0.35
    return wall_clock_time_seconds, generation_time_seconds


def build_sample_stats(benchmark: str, result: dict) -> dict:
    prediction = str(result.get("prediction", "")).strip()
    output_tokens = estimate_output_tokens(prediction)
    wall_clock_time_seconds, generation_time_seconds = estimate_timing(benchmark, output_tokens)
    predicted_boxes = list(result.get("predicted_boxes", []))
    false_positive_count = int(result.get("fp", max(0, len(predicted_boxes) - len(result.get("matched_predictions", [])))))
    false_negative_count = int(
        result.get(
            "fn",
            max(0, len(result.get("ground_truth_boxes", [])) - len(result.get("matched_targets", []))),
        )
    )
    generated_output_count = len(predicted_boxes) if benchmark in DETECTION_BENCHMARKS else (1 if prediction else 0)
    cpu_ram_bytes = 4_800_000_000 + (25_000_000 * output_tokens)
    return {
        "success": True,
        "error": None,
        "wall_clock_time_seconds": wall_clock_time_seconds,
        "generation_time_seconds": generation_time_seconds,
        "first_token_latency_seconds": None,
        "time_per_output_token_seconds": (generation_time_seconds / output_tokens) if output_tokens else None,
        "output_tokens": output_tokens,
        "retry_count": 0,
        "truncated": output_tokens >= BENCHMARK_TOKENS.get(benchmark, 24),
        "peak_cpu_ram_bytes": cpu_ram_bytes,
        "cpu_utilization_percent": 185.0,
        "peak_gpu_memory_bytes": None,
        "gpu_utilization_percent": None,
        "vram_allocation_over_time_bytes": None,
        "disk_offload_volume_bytes": None,
        "generated_output_count": generated_output_count,
        "hallucinated_label_count": 0,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "predicted_detection_count": len(predicted_boxes),
        "stats_source": "estimated_from_legacy_result",
        "stats_measured": False,
    }


def build_report_stats(benchmark: str, results: list[dict]) -> dict:
    sample_stats = [item["stats"] for item in results]
    total_wall = sum(float(item["wall_clock_time_seconds"]) for item in sample_stats)
    total_generation = sum(float(item["generation_time_seconds"]) for item in sample_stats)
    output_tokens = [int(item["output_tokens"]) for item in sample_stats]
    completed_count = len(sample_stats)
    return {
        "wall_clock_time_seconds": total_wall,
        "wall_clock_time_per_sample_seconds_mean": total_wall / completed_count if completed_count else None,
        "total_generation_time_seconds_mean": total_generation / completed_count if completed_count else None,
        "first_token_latency_seconds_mean": None,
        "time_per_output_token_seconds_mean": (
            sum(
                item["time_per_output_token_seconds"]
                for item in sample_stats
                if item["time_per_output_token_seconds"] is not None
            )
            / max(1, sum(1 for item in sample_stats if item["time_per_output_token_seconds"] is not None))
        ),
        "samples_per_second": (completed_count / total_wall) if total_wall > 0 else None,
        "number_of_output_tokens_mean": (sum(output_tokens) / completed_count) if completed_count else None,
        "number_of_generated_outputs_mean": (
            sum(int(item["generated_output_count"]) for item in sample_stats) / completed_count if completed_count else None
        ),
        "number_of_benchmark_samples_completed": completed_count,
        "success_count": completed_count,
        "failure_count": 0,
        "retry_count": 0,
        "peak_cpu_ram_bytes": max(int(item["peak_cpu_ram_bytes"]) for item in sample_stats) if sample_stats else None,
        "peak_gpu_memory_bytes": None,
        "cpu_utilization_percent_mean": 185.0 if sample_stats else None,
        "gpu_utilization_percent_mean": None,
        "vram_allocation_over_time_bytes": None,
        "disk_offload_volume_bytes": None,
        "truncation_rate": sum(1 for item in sample_stats if item["truncated"]) / completed_count if completed_count else None,
        "hallucinated_label_rate": 0.0,
        "false_positive_count_total": sum(int(item["false_positive_count"]) for item in sample_stats),
        "false_negative_count_total": sum(int(item["false_negative_count"]) for item in sample_stats),
        "mean_number_of_predicted_detections": (
            sum(int(item["predicted_detection_count"]) for item in sample_stats) / completed_count if completed_count else None
        ),
        "tokens_per_second": (sum(output_tokens) / total_generation) if total_generation > 0 else None,
        "model_load_time_seconds": 11.8,
        "stats_source": "estimated_from_legacy_result",
        "stats_measured": False,
    }


def main() -> int:
    source_dir = Path("old_results")
    target_dir = Path("results")
    target_dir.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(source_dir.glob("qwen25-vl_*.json")):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        report = payload.setdefault("report", {})
        benchmark = str(payload.get("benchmark") or report.get("benchmark") or source_path.stem)
        results = list(report.get("results", []))
        for result in results:
            result["stats"] = build_sample_stats(benchmark=benchmark, result=result)
        report["stats"] = build_report_stats(benchmark=benchmark, results=results)
        target_path = target_dir / source_path.name
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(target_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
