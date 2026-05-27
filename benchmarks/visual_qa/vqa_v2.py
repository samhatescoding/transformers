from dataset import VQAv2

from ._visual_qa import VisualQABenchmark


class VQAv2Benchmark(VisualQABenchmark):
    dataset_cls = VQAv2
    benchmark_name = "vqav2"
    default_split = "validation"
