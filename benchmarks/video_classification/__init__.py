from ._video_classification import VideoClassificationBenchmark
from .dfdc import DFDCBenchmark
from .kinetics import KineticsBenchmark
from .ucf101 import UCF101Benchmark

__all__ = [
    "VideoClassificationBenchmark",
    "DFDCBenchmark",
    "KineticsBenchmark",
    "UCF101Benchmark",
]
