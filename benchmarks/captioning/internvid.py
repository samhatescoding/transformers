from dataset import InternVid

from ._captioning import CaptioningBenchmark


class InternVidBenchmark(CaptioningBenchmark):
    dataset_cls = InternVid
    benchmark_name = "internvid"
    default_split = "train"
    preview_preparation_message = (
        "Streaming InternVid metadata and grouping all captions by distinct video."
    )

    def prepare(self, n, label_sample_size):
        return super().prepare(n, label_sample_size)
