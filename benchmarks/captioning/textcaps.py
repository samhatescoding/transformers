from dataset import TextCaps

from ._captioning import CaptioningBenchmark


class TextCapsBenchmark(CaptioningBenchmark):
    benchmark_name = "textcaps"
    default_max_new_tokens = 24

    def __init__(self, dataset=None, bleu_threshold: float = 0.25, split: str = "validation", streaming: bool = True):
        dataset = dataset or TextCaps(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name, bleu_threshold=bleu_threshold)
