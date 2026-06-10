from ._multiple_choice import MultipleChoiceBenchmark
from .blip3o_60k import BLIP3o60kBenchmark
from .conceptual_captions import ConceptualCaptionsBenchmark
from .diffusiondb import DiffusionDBBenchmark
from .flyingthings3d import FlyingThings3DBenchmark
from .hq_edit import HQEditBenchmark
from .imgedit import ImgEditBenchmark
from .magicbrush import MagicBrushBenchmark
from .openvid1m import OpenVid1MBenchmark
from .pick_a_pic import PickAPicBenchmark
from .sharegpt4o_image import ShareGPT4oImageBenchmark
from .sharegpt4o_image_edit import ShareGPT4oImageEditBenchmark

__all__ = [
    "MultipleChoiceBenchmark",
    "BLIP3o60kBenchmark",
    "ConceptualCaptionsBenchmark",
    "DiffusionDBBenchmark",
    "FlyingThings3DBenchmark",
    "HQEditBenchmark",
    "ImgEditBenchmark",
    "MagicBrushBenchmark",
    "OpenVid1MBenchmark",
    "PickAPicBenchmark",
    "ShareGPT4oImageBenchmark",
    "ShareGPT4oImageEditBenchmark",
]
