from dataset import Kinetics

from ._video_classification import VideoClassificationBenchmark


class KineticsBenchmark(VideoClassificationBenchmark):
    benchmark_name = "kinetics"

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or Kinetics(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
