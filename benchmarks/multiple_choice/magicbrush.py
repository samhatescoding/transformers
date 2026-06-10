from dataset import MagicBrush

from ..image_modification_vqa import ImageModificationVQABenchmark


class MagicBrushBenchmark(ImageModificationVQABenchmark):
    dataset_cls = MagicBrush
    benchmark_name = "magicbrush"
    default_split = "train"
    fallback_distractors = ("remove the main object", "change the background", "add a new object")
