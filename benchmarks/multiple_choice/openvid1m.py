from dataset import OpenVid1M

from ._multiple_choice import MultipleChoiceBenchmark


class OpenVid1MBenchmark(MultipleChoiceBenchmark):
    dataset_cls = OpenVid1M
    benchmark_name = "openvid1m"
    default_split = "train"
    default_instruction = "Choose the prompt that best matches the video frames."
    fallback_distractors = ("a person cooking in a kitchen", "a dog running in a park", "a car driving down a road")
