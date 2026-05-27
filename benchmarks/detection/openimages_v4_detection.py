from dataset import OpenImagesV4

from ._detection import DetectionBenchmark


class OpenImagesV4DetectionBenchmark(DetectionBenchmark):
    dataset_cls = OpenImagesV4
    benchmark_name = "openimages_v4_detection"
    default_split = "validation"
