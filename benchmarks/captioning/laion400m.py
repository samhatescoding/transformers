from dataset import LAION400M

from ._captioning import CaptioningBenchmark


class LAION400MBenchmark(CaptioningBenchmark):
    dataset_cls = LAION400M
    benchmark_name = "laion400m"
    default_split = "train"
