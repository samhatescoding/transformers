from dataset import HDTF

from ._multiple_choice import MultipleChoiceBenchmark


class HDTFBenchmark(MultipleChoiceBenchmark):
    dataset_cls = HDTF
    benchmark_name = "hdtf"
    default_split = "train"
    default_instruction = "Choose the transcript or caption that best matches the talking-face clip."
    fallback_distractors = ("silent landscape footage", "a street scene", "a product photo")
