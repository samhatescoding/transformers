from dataset import DiffusionDB

from ._multiple_choice import MultipleChoiceBenchmark


class DiffusionDBBenchmark(MultipleChoiceBenchmark):
    dataset_cls = DiffusionDB
    benchmark_name = "diffusiondb"
    default_split = "train"
    default_instruction = "Choose the text-to-image prompt that best matches the generated image."
    fallback_distractors = ("a product catalog photo", "a street-view segmentation", "a handwritten invoice")
