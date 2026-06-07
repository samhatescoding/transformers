from ._base_benchmark import BaseBenchmark
from .captioning._captioning import CaptioningBenchmark
from .captioning.conceptual_captions_caption import ConceptualCaptionsCaptionBenchmark
from .captioning.flickr30k import Flickr30kBenchmark
from .captioning.mscoco_caption import MSCOCOCaptionBenchmark
from .captioning.textcaps import TextCapsBenchmark
from .classification._classification import ClassificationBenchmark
from .classification.cityscapes import CityscapesBenchmark
from .classification.fairface import FairFaceBenchmark
from .classification.fashion_mnist import FashionMNISTBenchmark
from .classification.imagenet1k import ImageNet1kBenchmark
from .classification.inaturalist import INaturalistBenchmark
from .classification.lsun import LSUNBenchmark
from .classification.mvtec_ad import MVTecADBenchmark
from .classification.openimages_v4 import OpenImagesV4Benchmark
from .classification.places import PlacesBenchmark
from .classification.tad66k import TAD66KBenchmark
from .detection._detection import DetectionBenchmark
from .detection.flickr30k_entities import Flickr30kEntitiesBenchmark
from .detection.lvis import LVISBenchmark
from .detection.mscoco import MSCOCOBenchmark
from .detection.openimages_v4_detection import OpenImagesV4DetectionBenchmark
from .multiple_choice._multiple_choice import MultipleChoiceBenchmark
from .multiple_choice.blip3o_60k import BLIP3o60kBenchmark
from .multiple_choice.conceptual_captions import ConceptualCaptionsBenchmark
from .multiple_choice.diffusiondb import DiffusionDBBenchmark
from .multiple_choice.flyingthings3d import FlyingThings3DBenchmark
from .multiple_choice.hdtf import HDTFBenchmark
from .multiple_choice.hq_edit import HQEditBenchmark
from .multiple_choice.imgedit import ImgEditBenchmark
from .multiple_choice.internvid import InternVidBenchmark
from .multiple_choice.laion400m import LAION400MBenchmark
from .multiple_choice.laion5b import LAION5BBenchmark
from .multiple_choice.magicbrush import MagicBrushBenchmark
from .multiple_choice.openvid1m import OpenVid1MBenchmark
from .multiple_choice.pick_a_pic import PickAPicBenchmark
from .multiple_choice.sharegpt4o_image import ShareGPT4oImageBenchmark
from .video_classification._video_classification import VideoClassificationBenchmark
from .video_classification.dfdc import DFDCBenchmark
from .video_classification.kinetics import KineticsBenchmark
from .video_classification.ucf101 import UCF101Benchmark
from .visual_qa._visual_qa import VisualQABenchmark
from .visual_qa.docvqa import DocVQABenchmark
from .visual_qa.gqa import GQABenchmark
from .visual_qa.visual_cot import VisualCoTBenchmark
from .visual_qa.visual_genome import VisualGenomeBenchmark
from .visual_qa.vqa_v2 import VQAv2Benchmark

__all__ = [
    "BaseBenchmark",
    "CaptioningBenchmark",
    "ClassificationBenchmark",
    "DetectionBenchmark",
    "MultipleChoiceBenchmark",
    "VideoClassificationBenchmark",
    "VisualQABenchmark",
    "BLIP3o60kBenchmark",
    "ConceptualCaptionsBenchmark",
    "ConceptualCaptionsCaptionBenchmark",
    "CityscapesBenchmark",
    "DFDCBenchmark",
    "DiffusionDBBenchmark",
    "DocVQABenchmark",
    "FairFaceBenchmark",
    "FashionMNISTBenchmark",
    "Flickr30kBenchmark",
    "Flickr30kEntitiesBenchmark",
    "FlyingThings3DBenchmark",
    "GQABenchmark",
    "HDTFBenchmark",
    "HQEditBenchmark",
    "ImageNet1kBenchmark",
    "ImgEditBenchmark",
    "INaturalistBenchmark",
    "InternVidBenchmark",
    "KineticsBenchmark",
    "LAION400MBenchmark",
    "LAION5BBenchmark",
    "LSUNBenchmark",
    "LVISBenchmark",
    "MSCOCOBenchmark",
    "MSCOCOCaptionBenchmark",
    "MVTecADBenchmark",
    "MagicBrushBenchmark",
    "OpenImagesV4Benchmark",
    "OpenImagesV4DetectionBenchmark",
    "OpenVid1MBenchmark",
    "PickAPicBenchmark",
    "PlacesBenchmark",
    "ShareGPT4oImageBenchmark",
    "TAD66KBenchmark",
    "TextCapsBenchmark",
    "UCF101Benchmark",
    "VisualCoTBenchmark",
    "VisualGenomeBenchmark",
    "VQAv2Benchmark",
]
