from dataset import ConceptualCaptions

from ._captioning import CaptioningBenchmark


class ConceptualCaptionsCaptionBenchmark(CaptioningBenchmark):
    dataset_cls = ConceptualCaptions
    benchmark_name = "conceptual_captions_caption"
    default_split = "validation"
