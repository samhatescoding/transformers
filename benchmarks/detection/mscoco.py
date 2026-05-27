from dataset import MSCOCO

from ._detection import DetectionBenchmark


class MSCOCOBenchmark(DetectionBenchmark):
    dataset_cls = MSCOCO
    benchmark_name = "mscoco"
    default_split = "validation"
