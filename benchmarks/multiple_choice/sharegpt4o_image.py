from dataset import ShareGPT4oImage

from ..prompt_reconstruction import PromptReconstructionBenchmark


class ShareGPT4oImageBenchmark(PromptReconstructionBenchmark):
    dataset_cls = ShareGPT4oImage
    benchmark_name = "sharegpt4o_image"
    default_split = "train"
    fallback_distractors = (
        "a product photograph on a white background",
        "a handwritten note on a wooden desk",
        "a mountain landscape during sunrise",
    )
