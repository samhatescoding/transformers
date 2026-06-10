from dataset.sharegpt4o_image import ShareGPT4oImageEdit

from ..image_modification_vqa import ImageModificationVQABenchmark


class ShareGPT4oImageEditBenchmark(ImageModificationVQABenchmark):
    dataset_cls = ShareGPT4oImageEdit
    benchmark_name = "sharegpt4o_image_edit"
    default_split = "train"
    fallback_distractors = (
        "remove the main subject from the image",
        "convert the image into a monochrome sketch",
        "replace the background with a mountain landscape",
    )
