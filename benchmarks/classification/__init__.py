from ._classification import ClassificationBenchmark
from .cityscapes import CityscapesBenchmark
from .fairface import FairFaceBenchmark
from .fashion_mnist import FashionMNISTBenchmark
from .imagenet1k import ImageNet1kBenchmark
from .inaturalist import INaturalistBenchmark
from .lsun import LSUNBenchmark
from .mvtec_ad import MVTecADBenchmark
from .openimages_v4 import OpenImagesV4Benchmark
from .places import PlacesBenchmark
from .tad66k import TAD66KBenchmark

__all__ = [
    "ClassificationBenchmark",
    "CityscapesBenchmark",
    "FairFaceBenchmark",
    "FashionMNISTBenchmark",
    "ImageNet1kBenchmark",
    "INaturalistBenchmark",
    "LSUNBenchmark",
    "MVTecADBenchmark",
    "OpenImagesV4Benchmark",
    "PlacesBenchmark",
    "TAD66KBenchmark",
]
