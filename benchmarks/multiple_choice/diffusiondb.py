from dataset import DiffusionDB

from ..prompt_reconstruction import PromptReconstructionBenchmark


class DiffusionDBBenchmark(PromptReconstructionBenchmark):
    dataset_cls = DiffusionDB
    benchmark_name = "diffusiondb"
    default_split = "train"
    fallback_distractors = ("a product catalog photo", "a street-view segmentation", "a handwritten invoice")
