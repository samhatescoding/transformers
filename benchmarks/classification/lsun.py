from dataset import LSUN

from ._classification import ClassificationBenchmark


class LSUNBenchmark(ClassificationBenchmark):
    benchmark_name = "lsun"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or LSUN(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
