from dataset import ConceptualCaptions

from ._multiple_choice import MultipleChoiceBenchmark


class ConceptualCaptionsBenchmark(MultipleChoiceBenchmark):
    benchmark_name = "conceptual_captions"
    default_max_new_tokens = 16
    default_instruction = "Choose the caption that best matches the image."
    fallback_distractors = ("A city street at night", "A close-up of a person", "An animal in a field")

    def __init__(self, dataset=None, split: str = "validation", streaming: bool = True):
        dataset = dataset or ConceptualCaptions(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
