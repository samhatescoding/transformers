from dataset import VisualCoT

from ._visual_qa import VisualQABenchmark


class VisualCoTBenchmark(VisualQABenchmark):
    benchmark_name = "visual_cot"

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or VisualCoT(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
