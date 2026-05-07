from ._detection import DetectionBenchmark
from .flickr30k_entities import Flickr30kEntitiesBenchmark
from .lvis import LVISBenchmark
from .mscoco import MSCOCOBenchmark
from .openimages_v4_detection import OpenImagesV4DetectionBenchmark

__all__ = [
    "DetectionBenchmark",
    "Flickr30kEntitiesBenchmark",
    "LVISBenchmark",
    "MSCOCOBenchmark",
    "OpenImagesV4DetectionBenchmark",
]
