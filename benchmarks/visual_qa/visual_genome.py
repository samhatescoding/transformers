from dataset import VisualGenome

from ._visual_qa import VisualQABenchmark


class VisualGenomeBenchmark(VisualQABenchmark):
    dataset_cls = VisualGenome
    benchmark_name = "visual_genome"
    default_split = "train"
