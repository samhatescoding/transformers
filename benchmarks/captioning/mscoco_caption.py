from dataset import MSCOCOCaption

from ._captioning import CaptioningBenchmark


class MSCOCOCaptionBenchmark(CaptioningBenchmark):
    dataset_cls = MSCOCOCaption
    benchmark_name = "mscoco_caption"
    default_split = "test"
    default_max_new_tokens = 32
