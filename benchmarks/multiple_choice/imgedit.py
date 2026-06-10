from dataset import ImgEdit

from ..image_modification_vqa import ImageModificationVQABenchmark


class ImgEditBenchmark(ImageModificationVQABenchmark):
    dataset_cls = ImgEdit
    benchmark_name = "imgedit"
    default_split = "train"
    fallback_distractors = ("classify the scene", "read the document", "describe video motion")
