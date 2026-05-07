from dataset import Flickr30kEntities

from ._detection import DetectionBenchmark


class Flickr30kEntitiesBenchmark(DetectionBenchmark):
    benchmark_name = "flickr30k_entities"
    default_max_new_tokens = 32

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or Flickr30kEntities(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
