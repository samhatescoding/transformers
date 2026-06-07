from dataset import FlyingThings3D

from ._multiple_choice import MultipleChoiceBenchmark


class FlyingThings3DBenchmark(MultipleChoiceBenchmark):
    dataset_cls = FlyingThings3D
    benchmark_name = "flyingthings3d"
    default_split = "train"
    default_instruction = "Choose the description that best matches the rendered stereo scene."
    fallback_distractors = ("a close-up face video", "a document page", "a city street segmentation")
