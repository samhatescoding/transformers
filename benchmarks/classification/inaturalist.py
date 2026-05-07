from dataset import INaturalist

from ._classification import ClassificationBenchmark


class INaturalistBenchmark(ClassificationBenchmark):
    benchmark_name = "inaturalist"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or INaturalist(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
