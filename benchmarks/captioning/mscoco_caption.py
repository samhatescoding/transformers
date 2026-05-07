from dataset import MSCOCOCaption

from ._captioning import CaptioningBenchmark


class MSCOCOCaptionBenchmark(CaptioningBenchmark):
    benchmark_name = "mscoco_caption"
    default_max_new_tokens = 32

    def __init__(self, dataset=None, split: str = "test", streaming: bool = True, bleu_threshold: float = 0.25):
        dataset = dataset or MSCOCOCaption(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name, bleu_threshold=bleu_threshold)
