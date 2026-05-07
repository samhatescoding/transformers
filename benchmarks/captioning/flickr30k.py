from dataset import Flickr30k

from ._captioning import CaptioningBenchmark


class Flickr30kBenchmark(CaptioningBenchmark):
    benchmark_name = "flickr30k"
    default_max_new_tokens = 24

    def __init__(self, dataset=None, split: str = "test", streaming: bool = True, bleu_threshold: float = 0.25):
        dataset = dataset or Flickr30k(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name, bleu_threshold=bleu_threshold)
