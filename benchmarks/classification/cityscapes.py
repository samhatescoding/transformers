from dataset import Cityscapes

from ._classification import ClassificationBenchmark


class CityscapesBenchmark(ClassificationBenchmark):
    dataset_cls = Cityscapes
    benchmark_name = "cityscapes"
    default_split = "train"
