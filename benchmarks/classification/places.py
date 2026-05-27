from dataset import Places

from ._classification import ClassificationBenchmark


class PlacesBenchmark(ClassificationBenchmark):
    dataset_cls = Places
    benchmark_name = "places"
    default_split = "validation"
