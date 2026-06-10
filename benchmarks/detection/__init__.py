from ._detection import DetectionBenchmark
from .flickr30k_entities import Flickr30kEntitiesBenchmark
from .inaturalist import INaturalistDetectionBenchmark
from .lvis import LVISBenchmark
from .mscoco import MSCOCOBenchmark
from .openimages_v4_detection import OpenImagesV4DetectionBenchmark
from .visual_cot import VisualCoTDetectionBenchmark

__all__ = [
    "DetectionBenchmark",
    "Flickr30kEntitiesBenchmark",
    "INaturalistDetectionBenchmark",
    "LVISBenchmark",
    "MSCOCOBenchmark",
    "OpenImagesV4DetectionBenchmark",
    "VisualCoTDetectionBenchmark",
]
