from dataset import ShareGPT4oImage

from ._multiple_choice import MultipleChoiceBenchmark


class ShareGPT4oImageBenchmark(MultipleChoiceBenchmark):
    dataset_cls = ShareGPT4oImage
    benchmark_name = "sharegpt4o_image"
    default_split = "train"
    default_instruction = "Choose the prompt or instruction that best matches the generated target image."
    fallback_distractors = ("detect rare objects", "classify an action video", "segment a road scene")
