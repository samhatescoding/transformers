from dataset import LAION400M

from ._multiple_choice import MultipleChoiceBenchmark


class LAION400MBenchmark(MultipleChoiceBenchmark):
    benchmark_name = "laion400m"
    default_instruction = "Choose the caption that best matches the image."
    fallback_distractors = ("A landscape photo", "A portrait of a person", "A product shot on a table")

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or LAION400M(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
