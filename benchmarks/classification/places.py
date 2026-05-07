from dataset import Places

from ._classification import ClassificationBenchmark


class PlacesBenchmark(ClassificationBenchmark):
    benchmark_name = "places"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or Places(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
