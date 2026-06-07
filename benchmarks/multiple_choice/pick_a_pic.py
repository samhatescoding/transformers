from dataset import PickAPic

from ._multiple_choice import MultipleChoiceBenchmark


class PickAPicBenchmark(MultipleChoiceBenchmark):
    dataset_cls = PickAPic
    benchmark_name = "pick_a_pic"
    default_split = "train"
    default_instruction = "Choose the prompt that best matches the preferred generated image."
    fallback_distractors = ("a low-resolution video clip", "a document scan", "a semantic segmentation mask")
