from dataset import LVIS

from .._base_benchmark import BaseBenchmark
from .mscoco import MSCOCOBenchmark


class LVISBenchmark(MSCOCOBenchmark):
    benchmark_name = "lvis"

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or LVIS(split=split, streaming=streaming)
        BaseBenchmark.__init__(self, dataset=dataset, name=self.benchmark_name)
