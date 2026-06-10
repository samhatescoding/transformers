from dataset import LVIS

from ._detection import DetectionBenchmark


class LVISBenchmark(DetectionBenchmark):
    dataset_cls = LVIS
    benchmark_name = "lvis"
    default_split = "validation"
