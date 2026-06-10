from dataset import HQEdit

from ..image_modification_vqa import ImageModificationVQABenchmark


class HQEditBenchmark(ImageModificationVQABenchmark):
    dataset_cls = HQEdit
    benchmark_name = "hq_edit"
    default_split = "train"
    fallback_distractors = ("make no change", "remove all foreground objects", "convert the image to a document")
