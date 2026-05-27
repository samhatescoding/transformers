from dataset import Flickr30k

from ._captioning import CaptioningBenchmark


class Flickr30kBenchmark(CaptioningBenchmark):
    dataset_cls = Flickr30k
    benchmark_name = "flickr30k"
    default_split = "test"
