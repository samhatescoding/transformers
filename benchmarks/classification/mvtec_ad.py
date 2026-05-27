from dataset import MVTecAD

from ._classification import ClassificationBenchmark


class MVTecADBenchmark(ClassificationBenchmark):
    dataset_cls = MVTecAD
    benchmark_name = "mvtec_ad"
    default_split = "test"
