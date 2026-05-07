from dataset import OpenImagesV4

from ._detection import DetectionBenchmark


class OpenImagesV4DetectionBenchmark(DetectionBenchmark):
    benchmark_name = "openimages_v4_detection"
    default_max_new_tokens = 32

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or OpenImagesV4(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
