from dataset import FashionMNIST

from ._classification import ClassificationBenchmark


class FashionMNISTBenchmark(ClassificationBenchmark):
    dataset_cls = FashionMNIST
    benchmark_name = "fashion_mnist"
    default_split = "train"
