from dataset import GQA

from ._visual_qa import VisualQABenchmark


class GQABenchmark(VisualQABenchmark):
    dataset_cls = GQA
    benchmark_name = "gqa"
    default_split = "validation"
