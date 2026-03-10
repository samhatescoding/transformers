import traceback

from benchmarks import MSCOCOBenchmark, Flickr30kBenchmark
from models import Llava
from ui import BenchmarkUI, BenchmarkSampleSaver


# Edit these values directly to control benchmark size.
TEST_SIZE = 2
LABEL_SAMPLE_SIZE = 10
ENABLE_UI = True
SAVE_SAMPLE_IMAGES = True
SAMPLE_OUTPUT_DIR = "ui_outputs"
KEEP_UI_OPEN_AFTER_RUN = True


def main() -> int:
    test_size = max(1, TEST_SIZE)
    label_sample_size = max(1, LABEL_SAMPLE_SIZE)
    viewer = BenchmarkUI(title="Benchmark Sample Viewer") if ENABLE_UI else None
    saver = BenchmarkSampleSaver(base_dir=SAMPLE_OUTPUT_DIR) if SAVE_SAMPLE_IMAGES else None

    if ENABLE_UI and viewer is not None and not viewer.enabled:
        print("[WARN] UI is enabled but unavailable in this environment. Continuing without UI.")
    if saver is not None:
        print(f"[INFO] Saving annotated samples to: {saver.output_dir}")

    def on_sample(payload):
        if viewer and viewer.enabled:
            viewer.on_sample(payload)
        if saver:
            saver.on_sample(payload)

    try:
        benchmark = MSCOCOBenchmark()
        model = Llava(max_new_tokens=16)

        report = benchmark.run(
            model=model,
            n=test_size,
            label_sample_size=label_sample_size,
            show_progress=True,
            on_sample=on_sample if (viewer and viewer.enabled) or saver else None,
        )
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

    print(f"Benchmark: {report['benchmark']}")
    print(f"Dataset: {report['dataset']}")
    print(f"Evaluated {report['num_samples']} samples")
    print(f"Candidate labels: {report['num_candidate_labels']}")

    correct = 0
    for item in report["results"]:
        print(f"\nImage {item['index']}/{report['num_samples']}")
        print("Prediction:", item["prediction"])
        if item["correct"]:
            correct += 1
            print("Correct")
        else:
            print("Incorrect")
            print("Valid labels for this image:", item["valid_labels"])

    print(f"\nAccuracy: {correct}/{report['num_samples']}")

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
