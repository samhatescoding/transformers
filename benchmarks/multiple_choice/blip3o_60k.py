from dataset import BLIP3o60k

from ._multiple_choice import MultipleChoiceBenchmark


class BLIP3o60kBenchmark(MultipleChoiceBenchmark):
    benchmark_name = "blip3o_60k"
    default_instruction = "Choose the editing instruction that best explains the change."
    fallback_distractors = ("add an object", "remove the object", "change the style")

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or BLIP3o60k(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
