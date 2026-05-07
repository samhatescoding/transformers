from dataset import MVTecAD

from ._classification import ClassificationBenchmark


class MVTecADBenchmark(ClassificationBenchmark):
    benchmark_name = "mvtec_ad"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "test", streaming: bool = True):
        dataset = dataset or MVTecAD(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
