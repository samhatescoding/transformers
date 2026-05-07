from dataset import DFDC

from ._video_classification import VideoClassificationBenchmark


class DFDCBenchmark(VideoClassificationBenchmark):
    benchmark_name = "dfdc"
    default_max_new_tokens = 16

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or DFDC(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
