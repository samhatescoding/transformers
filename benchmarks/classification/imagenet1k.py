from dataset import ImageNet1k

from ._classification import ClassificationBenchmark


class ImageNet1kBenchmark(ClassificationBenchmark):
    dataset_cls = ImageNet1k
    benchmark_name = "imagenet1k"
    default_split = "validation"
