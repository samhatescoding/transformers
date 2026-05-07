from dataset import LAION5B

from ._multiple_choice import MultipleChoiceBenchmark


class LAION5BBenchmark(MultipleChoiceBenchmark):
    benchmark_name = "laion5b"
    default_instruction = "Choose the caption that best matches the image."
    fallback_distractors = ("A detailed studio portrait", "A city skyline", "A food close-up")

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or LAION5B(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
