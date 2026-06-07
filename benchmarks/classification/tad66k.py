from dataset import TAD66K

from ._classification import ClassificationBenchmark


class TAD66KBenchmark(ClassificationBenchmark):
    dataset_cls = TAD66K
    benchmark_name = "tad66k"
    default_split = "train"
