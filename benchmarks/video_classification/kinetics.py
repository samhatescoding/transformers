from dataset import Kinetics

from ._video_classification import VideoClassificationBenchmark


class KineticsBenchmark(VideoClassificationBenchmark):
    dataset_cls = Kinetics
    benchmark_name = "kinetics"
    default_split = "train"
