from dataset import ConceptualCaptions

from ._captioning import CaptioningBenchmark


class ConceptualCaptionsCaptionBenchmark(CaptioningBenchmark):
    benchmark_name = "conceptual_captions_caption"
    default_max_new_tokens = 24

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True, bleu_threshold: float = 0.25):
        dataset = dataset or ConceptualCaptions(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name, bleu_threshold=bleu_threshold)
