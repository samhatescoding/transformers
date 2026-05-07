from ._visual_qa import VisualQABenchmark
from .docvqa import DocVQABenchmark
from .gqa import GQABenchmark
from .visual_cot import VisualCoTBenchmark
from .visual_genome import VisualGenomeBenchmark
from .vqa_v2 import VQAv2Benchmark

__all__ = [
    "VisualQABenchmark",
    "DocVQABenchmark",
    "GQABenchmark",
    "VisualCoTBenchmark",
    "VisualGenomeBenchmark",
    "VQAv2Benchmark",
]
