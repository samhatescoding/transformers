from dataset import BLIP3o60k

from ..prompt_reconstruction import PromptReconstructionBenchmark


class BLIP3o60kBenchmark(PromptReconstructionBenchmark):
    dataset_cls = BLIP3o60k
    benchmark_name = "blip3o_60k"
    default_split = "train"
    fallback_distractors = (
        "a studio photograph of a household object",
        "a watercolor landscape at sunset",
        "a busy city street at night",
    )
