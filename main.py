import json
import re
import traceback
from pathlib import Path

from benchmarks import (
    ImageNet1kBenchmark,
    INaturalistBenchmark,
    MSCOCOBenchmark,
    UCF101Benchmark,
    VQAv2Benchmark,
)
from models import SmallLlava
from models.llava import Llava
from ui import BenchmarkUI, BenchmarkSampleSaver


# Edit these values directly to control benchmark size.
TEST_SIZE = 10
LABEL_SAMPLE_SIZE = 16
ENABLE_UI = False
SAVE_SAMPLE_IMAGES = False
SAMPLE_OUTPUT_DIR = "ui/ui_outputs"
DELETE_OLD_OUTPUTS_ON_START = True
KEEP_UI_OPEN_AFTER_RUN = False
LOAD_MODEL_IN_4BIT = True
RESULTS_DIR = "results"
BENCHMARK_FACTORIES = [
    ImageNet1kBenchmark,
    INaturalistBenchmark,
    MSCOCOBenchmark,
    UCF101Benchmark,
    VQAv2Benchmark,
]


def _slugify_filename_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or "unknown"


def _save_report(report: dict, model_name: str, benchmark_name: str) -> Path:
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_part = _slugify_filename_part(model_name)
    benchmark_part = _slugify_filename_part(benchmark_name)
    output_path = output_dir / f"{model_part}_{benchmark_part}.json"

    payload = {
        "model": model_name,
        "benchmark": benchmark_name,
        "report": report,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _safe_console_text(value) -> str:
    text = str(value)
    return text.encode("ascii", "backslashreplace").decode("ascii")


def main() -> int:
    test_size = max(1, TEST_SIZE)
    label_sample_size = max(1, LABEL_SAMPLE_SIZE)
    viewer = BenchmarkUI(title="Benchmark Sample Viewer") if ENABLE_UI else None
    saver = (
        BenchmarkSampleSaver(
            base_dir=SAMPLE_OUTPUT_DIR,
            clear_existing_runs=DELETE_OLD_OUTPUTS_ON_START,
        )
        if SAVE_SAMPLE_IMAGES
        else None
    )

    if ENABLE_UI and viewer is not None and not viewer.enabled:
        print("[WARN] UI is enabled but unavailable in this environment. Continuing without UI.")
    if saver is not None:
        if DELETE_OLD_OUTPUTS_ON_START:
            print(f"[INFO] Cleared previous outputs in: {SAMPLE_OUTPUT_DIR}")
        print(f"[INFO] Saving annotated samples to: {saver.output_dir}")

    def on_sample(payload):
        if viewer and viewer.enabled:
            viewer.on_sample(payload)
        if saver:
            saver.on_sample(payload)

    try:
        model = Llava(max_new_tokens=64, load_in_4bit=LOAD_MODEL_IN_4BIT, stream=False)
        benchmark_reports = []
        benchmark_failures = []
        for benchmark_factory in BENCHMARK_FACTORIES:
            try:
                benchmark = benchmark_factory()
                print(f"\n[INFO] Running benchmark: {benchmark.name}")
                report = benchmark.run(
                    model=model,
                    n=test_size,
                    label_sample_size=label_sample_size,
                    show_progress=True,
                    on_sample=on_sample if (viewer and viewer.enabled) or saver else None,
                )
                results_path = _save_report(
                    report=report,
                    model_name=getattr(model, "name", model.__class__.__name__),
                    benchmark_name=getattr(benchmark, "name", benchmark.__class__.__name__),
                )
                benchmark_reports.append((benchmark, report, results_path))
            except Exception as benchmark_exc:
                benchmark_failures.append((benchmark.name, benchmark_exc, traceback.format_exc()))
                print(f"[ERROR] {benchmark.name} failed: {benchmark_exc.__class__.__name__}: {benchmark_exc}")
    except Exception as exc:
        print("\n[ERROR] Benchmark execution failed.")
        print(f"Reason: {exc.__class__.__name__}: {exc}")
        print("Full traceback:")
        print(traceback.format_exc())
        if viewer:
            viewer.close()
        if saver:
            saver.close()
        return 1

    for benchmark, report, results_path in benchmark_reports:
        print(f"\nBenchmark: {report['benchmark']}")
        print(f"Results file: {results_path}")
        print(f"Dataset: {report['dataset']}")
        print(f"Evaluated {report['num_samples']} samples")
        print(f"Dataset label pool: {report['num_candidate_labels']}")

        correct = 0
        for item in report["results"]:
            print(f"\nSample {item['index']}/{report['num_samples']}")
            prompt_labels = item.get("prompt_labels", [])
            if prompt_labels:
                print("Prompt labels:", _safe_console_text(", ".join(prompt_labels)))
            print("Prediction:", _safe_console_text(item["prediction"]))
            if item["correct"]:
                correct += 1
                print("Correct")
            else:
                print("Incorrect")
                print("Valid labels for this sample:", _safe_console_text(item["valid_labels"]))

            pred_boxes = item.get("predicted_boxes", [])
            gt_boxes = item.get("ground_truth_boxes", [])
            if pred_boxes or gt_boxes:
                print(f"Predicted boxes: {len(pred_boxes)}")
                for pred_idx, pred_box in enumerate(pred_boxes, start=1):
                    label = pred_box.get("label", "").strip() or "(no label)"
                    print(f"  {pred_idx}. {label} -> {pred_box.get('xyxy', [])}")
                print(f"Ground-truth boxes: {len(gt_boxes)}")

            if "matched_predictions" in item:
                print(
                    "Metrics:",
                    f"precision={item.get('precision', 0.0):.3f}",
                    f"recall={item.get('recall', 0.0):.3f}",
                    f"f1={item.get('f1', 0.0):.3f}",
                    f"mean_iou_matched={item.get('mean_iou_matched', 0.0):.3f}",
                    f"mean_iou_all_predictions={item.get('mean_iou_all_predictions', 0.0):.3f}",
                )
            if "point_accuracy" in item:
                print(f"Point accuracy: {item.get('point_accuracy', 0.0):.3f}")
            if "mean_endpoint_error" in item:
                print(f"Mean endpoint error: {item.get('mean_endpoint_error')}")

        print(f"\nAccuracy: {correct}/{report['num_samples']}")

    for benchmark_name, benchmark_exc, benchmark_traceback in benchmark_failures:
        print(f"\nBenchmark failed: {benchmark_name}")
        print(f"Reason: {benchmark_exc.__class__.__name__}: {benchmark_exc}")
        print(benchmark_traceback)

    if viewer and viewer.enabled and KEEP_UI_OPEN_AFTER_RUN:
        print("\n[INFO] Close the Sample Viewer window when you're done reviewing results.")
        viewer.wait_until_closed()

    if viewer:
        viewer.close()
    if saver:
        saver.close()

    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print(f"\nProgram finished with exit code {exit_code}.")
