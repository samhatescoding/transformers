import traceback

from benchmarks import MSCOCOBenchmark, Flickr30kBenchmark
from models import Llava


# Edit these values directly to control benchmark size.
TEST_SIZE = 2
LABEL_SAMPLE_SIZE = 10


def main() -> int:
    test_size = max(1, TEST_SIZE)
    label_sample_size = max(1, LABEL_SAMPLE_SIZE)

    try:
        benchmark = MSCOCOBenchmark()
        model = Llava(max_new_tokens=16)

        report = benchmark.run(
            model=model,
            n=test_size,
            label_sample_size=label_sample_size,
            show_progress=True,
        )
    except Exception as exc:
        print("\n[ERROR] Benchmark execution failed.")
        print(f"Reason: {exc.__class__.__name__}: {exc}")
        print("Full traceback:")
        print(traceback.format_exc())
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
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print(f"\nProgram finished with exit code {exit_code}.")
