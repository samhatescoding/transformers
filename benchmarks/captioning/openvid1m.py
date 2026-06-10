from dataset import OpenVid1M

from ._captioning import CaptioningBenchmark


class OpenVid1MCaptionBenchmark(CaptioningBenchmark):
    dataset_cls = OpenVid1M
    benchmark_name = "openvid1m_caption"
    default_split = "train"
