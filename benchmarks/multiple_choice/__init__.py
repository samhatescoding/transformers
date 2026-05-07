from ._multiple_choice import MultipleChoiceBenchmark
from .blip3o_60k import BLIP3o60kBenchmark
from .conceptual_captions import ConceptualCaptionsBenchmark
from .internvid import InternVidBenchmark
from .laion400m import LAION400MBenchmark
from .laion5b import LAION5BBenchmark
from .openvid1m import OpenVid1MBenchmark

__all__ = [
    "MultipleChoiceBenchmark",
    "BLIP3o60kBenchmark",
    "ConceptualCaptionsBenchmark",
    "InternVidBenchmark",
    "LAION400MBenchmark",
    "LAION5BBenchmark",
    "OpenVid1MBenchmark",
]
