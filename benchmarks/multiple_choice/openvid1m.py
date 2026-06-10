from dataset import OpenVid1M

from ..prompt_reconstruction import PromptReconstructionBenchmark


class OpenVid1MBenchmark(PromptReconstructionBenchmark):
    dataset_cls = OpenVid1M
    benchmark_name = "openvid1m"
    default_split = "train"
    fallback_distractors = ("a person cooking in a kitchen", "a dog running in a park", "a car driving down a road")
