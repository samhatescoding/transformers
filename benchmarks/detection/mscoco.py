from dataset import MSCOCO

from ._detection import DetectionBenchmark


class MSCOCOBenchmark(DetectionBenchmark):
    dataset_cls = MSCOCO
    benchmark_name = "mscoco"
    default_split = "validation"
    preview_preparation_message = (
        "Preparing HF rows and loading official COCO detection annotations. "
        "The first run may download a roughly 253 MB archive; later runs use "
        "the repository cache."
    )

    def prepare(self, n, label_sample_size):
        rows, labels = super().prepare(n=n, label_sample_size=label_sample_size)
        ensure_annotations_loaded = getattr(self.dataset, "_ensure_annotations_loaded", None)
        if callable(ensure_annotations_loaded):
            ensure_annotations_loaded()
        return rows, labels
