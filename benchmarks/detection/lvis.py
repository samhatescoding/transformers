from dataset import LVIS

from .mscoco import MSCOCOBenchmark


class LVISBenchmark(MSCOCOBenchmark):
    dataset_cls = LVIS
    benchmark_name = "lvis"
    default_split = "validation"
