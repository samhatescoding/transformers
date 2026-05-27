from dataset import LAION400M

from ._multiple_choice import MultipleChoiceBenchmark


class LAION400MBenchmark(MultipleChoiceBenchmark):
    dataset_cls = LAION400M
    benchmark_name = "laion400m"
    default_split = "train"
    default_instruction = "Choose the caption that best matches the image."
    fallback_distractors = ("A landscape photo", "A portrait of a person", "A product shot on a table")
