from dataset import HDTF

from ._captioning import CaptioningBenchmark


class HDTFBenchmark(CaptioningBenchmark):
    dataset_cls = HDTF
    benchmark_name = "hdtf"
    default_split = "train"
