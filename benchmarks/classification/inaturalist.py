from dataset import INaturalist

from ._classification import ClassificationBenchmark


class INaturalistBenchmark(ClassificationBenchmark):
    dataset_cls = INaturalist
    benchmark_name = "inaturalist"
    default_split = "train"
