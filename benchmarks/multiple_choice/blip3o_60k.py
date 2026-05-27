from dataset import BLIP3o60k

from ._multiple_choice import MultipleChoiceBenchmark


class BLIP3o60kBenchmark(MultipleChoiceBenchmark):
    dataset_cls = BLIP3o60k
    benchmark_name = "blip3o_60k"
    default_split = "train"
    default_instruction = "Choose the editing instruction that best explains the change."
    fallback_distractors = ("add an object", "remove the object", "change the style")
