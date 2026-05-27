from dataset import LAION5B

from ._multiple_choice import MultipleChoiceBenchmark


class LAION5BBenchmark(MultipleChoiceBenchmark):
    dataset_cls = LAION5B
    benchmark_name = "laion5b"
    default_split = "train"
    default_instruction = "Choose the caption that best matches the image."
    fallback_distractors = ("A detailed studio portrait", "A city skyline", "A food close-up")
