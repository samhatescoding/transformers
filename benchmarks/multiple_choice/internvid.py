from dataset import InternVid

from ._multiple_choice import MultipleChoiceBenchmark


class InternVidBenchmark(MultipleChoiceBenchmark):
    benchmark_name = "internvid"
    default_instruction = "Choose the caption that best matches the video frames."
    fallback_distractors = (
        "A person is speaking to camera",
        "An animal runs outdoors",
        "A vehicle moves through a street",
    )

    def __init__(self, dataset=None, split: str = "train", streaming: bool = True):
        dataset = dataset or InternVid(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
