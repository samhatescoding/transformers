from dataset import OpenVid1M

from ._multiple_choice import MultipleChoiceBenchmark


class OpenVid1MBenchmark(MultipleChoiceBenchmark):
    benchmark_name = "openvid1m"
    default_instruction = "Choose the prompt that best matches the video frames."
    fallback_distractors = ("a person cooking in a kitchen", "a dog running in a park", "a car driving down a road")

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or OpenVid1M(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
