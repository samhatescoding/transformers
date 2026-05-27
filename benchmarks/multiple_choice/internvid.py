from dataset import InternVid

from ._multiple_choice import MultipleChoiceBenchmark


class InternVidBenchmark(MultipleChoiceBenchmark):
    dataset_cls = InternVid
    benchmark_name = "internvid"
    default_split = "train"
    default_instruction = "Choose the caption that best matches the video frames."
    fallback_distractors = (
        "A person is speaking to camera",
        "An animal runs outdoors",
        "A vehicle moves through a street",
    )
