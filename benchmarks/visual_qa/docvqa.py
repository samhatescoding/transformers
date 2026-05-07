from dataset import DocVQA

from ._visual_qa import VisualQABenchmark


class DocVQABenchmark(VisualQABenchmark):
    benchmark_name = "docvqa"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or DocVQA(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
