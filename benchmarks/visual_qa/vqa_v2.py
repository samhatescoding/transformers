from dataset import VQAv2

from ._visual_qa import VisualQABenchmark


class VQAv2Benchmark(VisualQABenchmark):
    benchmark_name = "vqav2"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or VQAv2(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
