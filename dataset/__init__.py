from importlib import import_module

__all__ = [
    "BaseDataset",
    "BLIP3o60k",
    "ConceptualCaptions",
    "DFDC",
    "DocVQA",
    "FairFace",
    "FashionMNIST",
    "Flickr30k",
    "Flickr30kEntities",
    "GQA",
    "ImageNet1k",
    "INaturalist",
    "InternVid",
    "Kinetics",
    "LAION400M",
    "LAION5B",
    "LSUN",
    "LVIS",
    "MSCOCO",
    "MSCOCOCaption",
    "MVTecAD",
    "OpenImagesV4",
    "OpenVid1M",
    "Places",
    "TextCaps",
    "UCF101",
    "VisualCoT",
    "VisualGenome",
    "VQAv2",
]

_LAZY_IMPORTS = {
    "BaseDataset": "_base_dataset",
    "BLIP3o60k": "blip3o_60k",
    "ConceptualCaptions": "conceptual_captions",
    "DFDC": "dfdc",
    "DocVQA": "docvqa",
    "FairFace": "fairface",
    "FashionMNIST": "fashion_mnist",
    "Flickr30k": "flickr30k",
    "Flickr30kEntities": "flickr30k_entities",
    "GQA": "gqa",
    "ImageNet1k": "imagenet1k",
    "INaturalist": "inaturalist",
    "InternVid": "internvid",
    "Kinetics": "kinetics",
    "LAION400M": "laion400m",
    "LAION5B": "laion5b",
    "LSUN": "lsun",
    "LVIS": "lvis",
    "MSCOCO": "mscoco",
    "MSCOCOCaption": "mscoco_caption",
    "MVTecAD": "mvtec_ad",
    "OpenImagesV4": "openimages_v4",
    "OpenVid1M": "openvid1m",
    "Places": "places",
    "TextCaps": "textcaps",
    "UCF101": "ucf101",
    "VisualCoT": "visual_cot",
    "VisualGenome": "visual_genome",
    "VQAv2": "vqa_v2",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
