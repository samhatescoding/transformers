from dataset import ImgEdit

from ._multiple_choice import MultipleChoiceBenchmark


class ImgEditBenchmark(MultipleChoiceBenchmark):
    dataset_cls = ImgEdit
    benchmark_name = "imgedit"
    default_split = "train"
    default_instruction = "Choose the image-edit instruction that best explains the change."
    fallback_distractors = ("classify the scene", "read the document", "describe video motion")
