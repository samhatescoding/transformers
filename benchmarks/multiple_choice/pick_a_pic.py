from dataset import PickAPic

from ..image_preference import ImagePreferenceBenchmark


class PickAPicBenchmark(ImagePreferenceBenchmark):
    dataset_cls = PickAPic
    benchmark_name = "pick_a_pic"
    default_split = "train"
