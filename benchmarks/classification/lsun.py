from dataset import LSUN

from ._classification import ClassificationBenchmark


class LSUNBenchmark(ClassificationBenchmark):
    dataset_cls = LSUN
    benchmark_name = "lsun"
    default_split = "train"
