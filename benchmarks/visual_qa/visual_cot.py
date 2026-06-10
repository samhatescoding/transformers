from dataset import VisualCoT

from ._visual_qa import VisualQABenchmark


class VisualCoTBenchmark(VisualQABenchmark):
    dataset_cls = VisualCoT
    benchmark_name = "visual_cot"
    default_split = "train"
