from dataset import OpenImagesV4

from ._classification import ClassificationBenchmark


class OpenImagesV4Benchmark(ClassificationBenchmark):
    dataset_cls = OpenImagesV4
    benchmark_name = "openimages_v4"
    default_split = "train"
