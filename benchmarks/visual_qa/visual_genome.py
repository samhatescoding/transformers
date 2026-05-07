from dataset import VisualGenome

from ._visual_qa import VisualQABenchmark


class VisualGenomeBenchmark(VisualQABenchmark):
    benchmark_name = "visual_genome"

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or VisualGenome(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
