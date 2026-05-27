from dataset import TextCaps

from ._captioning import CaptioningBenchmark


class TextCapsBenchmark(CaptioningBenchmark):
    dataset_cls = TextCaps
    benchmark_name = "textcaps"
    default_split = "validation"
