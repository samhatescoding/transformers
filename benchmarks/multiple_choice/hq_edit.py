from dataset import HQEdit

from ._multiple_choice import MultipleChoiceBenchmark


class HQEditBenchmark(MultipleChoiceBenchmark):
    dataset_cls = HQEdit
    benchmark_name = "hq_edit"
    default_split = "train"
    default_instruction = "Choose the high-quality edit instruction that best explains the change."
    fallback_distractors = ("make no change", "remove all foreground objects", "convert the image to a document")
