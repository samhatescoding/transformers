from dataset.inaturalist_detection import INaturalistDetection

from ._detection import DetectionBenchmark


class INaturalistDetectionBenchmark(DetectionBenchmark):
    dataset_cls = INaturalistDetection
    benchmark_name = "inaturalist_detection"
    default_split = "train"
