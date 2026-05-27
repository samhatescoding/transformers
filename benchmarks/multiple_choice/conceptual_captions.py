from dataset import ConceptualCaptions

from ._multiple_choice import MultipleChoiceBenchmark


class ConceptualCaptionsBenchmark(MultipleChoiceBenchmark):
    dataset_cls = ConceptualCaptions
    benchmark_name = "conceptual_captions"
    default_split = "validation"
    default_instruction = "Choose the caption that best matches the image."
    fallback_distractors = ("A city street at night", "A close-up of a person", "An animal in a field")
