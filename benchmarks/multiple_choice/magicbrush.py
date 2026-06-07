from dataset import MagicBrush

from ._multiple_choice import MultipleChoiceBenchmark


class MagicBrushBenchmark(MultipleChoiceBenchmark):
    dataset_cls = MagicBrush
    benchmark_name = "magicbrush"
    default_split = "train"
    default_instruction = "Choose the editing instruction that best explains the change."
    fallback_distractors = ("remove the main object", "change the background", "add a new object")
