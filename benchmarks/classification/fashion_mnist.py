from dataset import FashionMNIST

from ._classification import ClassificationBenchmark


class FashionMNISTBenchmark(ClassificationBenchmark):
    benchmark_name = "fashion_mnist"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or FashionMNIST(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
