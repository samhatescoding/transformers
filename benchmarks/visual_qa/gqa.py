from dataset import GQA

from ._visual_qa import VisualQABenchmark


class GQABenchmark(VisualQABenchmark):
    benchmark_name = "gqa"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or GQA(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
