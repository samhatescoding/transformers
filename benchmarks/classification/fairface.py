from dataset import FairFace

from ._classification import ClassificationBenchmark


class FairFaceBenchmark(ClassificationBenchmark):
    dataset_cls = FairFace
    benchmark_name = "fairface"
    default_split = "train"
