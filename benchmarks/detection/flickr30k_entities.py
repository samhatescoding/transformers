from dataset import Flickr30kEntities

from ._detection import DetectionBenchmark


class Flickr30kEntitiesBenchmark(DetectionBenchmark):
    dataset_cls = Flickr30kEntities
    benchmark_name = "flickr30k_entities"
    default_split = "train"
