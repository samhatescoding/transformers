from dataset import OpenImagesV4

from ._classification import ClassificationBenchmark


class OpenImagesV4Benchmark(ClassificationBenchmark):
    benchmark_name = "openimages_v4"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or OpenImagesV4(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
