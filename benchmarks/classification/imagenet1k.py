from dataset import ImageNet1k

from ._classification import ClassificationBenchmark


class ImageNet1kBenchmark(ClassificationBenchmark):
    benchmark_name = "imagenet1k"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or ImageNet1k(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
