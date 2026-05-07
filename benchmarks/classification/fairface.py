from dataset import FairFace

from ._classification import ClassificationBenchmark


class FairFaceBenchmark(ClassificationBenchmark):
    benchmark_name = "fairface"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or FairFace(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
