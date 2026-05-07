from dataset import MSCOCO

from ._detection import DetectionBenchmark


class MSCOCOBenchmark(DetectionBenchmark):
    benchmark_name = "mscoco"
    default_max_new_tokens = 32

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or MSCOCO(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
