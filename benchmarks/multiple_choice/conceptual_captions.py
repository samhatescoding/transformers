from dataset import ConceptualCaptions

from ..prompt_reconstruction import PromptReconstructionBenchmark


class ConceptualCaptionsBenchmark(PromptReconstructionBenchmark):
    dataset_cls = ConceptualCaptions
    benchmark_name = "conceptual_captions"
    default_split = "validation"
    fallback_distractors = ("A city street at night", "A close-up of a person", "An animal in a field")
