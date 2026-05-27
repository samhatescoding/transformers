from dataset import DocVQA

from ._visual_qa import VisualQABenchmark


class DocVQABenchmark(VisualQABenchmark):
    dataset_cls = DocVQA
    benchmark_name = "docvqa"
    default_split = "validation"
