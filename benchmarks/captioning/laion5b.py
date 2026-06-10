from dataset import LAION5B

from ._captioning import CaptioningBenchmark


class LAION5BBenchmark(CaptioningBenchmark):
    dataset_cls = LAION5B
    benchmark_name = "laion5b"
    default_split = "train"
