from dataset import DFDC

from ._video_classification import VideoClassificationBenchmark


class DFDCBenchmark(VideoClassificationBenchmark):
    dataset_cls = DFDC
    benchmark_name = "dfdc"
    default_split = "train"
