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
    "BLIP3o60k": "caption_blip3o_60k",
    "ConceptualCaptions": "caption_conceptual_captions",
    "DFDC": "class_dfdc",
    "DocVQA": "qa_docvqa",
    "FairFace": "class_fairface",
    "FashionMNIST": "class_fashion_mnist",
    "Flickr30k": "caption_flickr30k",
    "Flickr30kEntities": "detect_flickr30k_entities",
    "GQA": "qa_gqa",
    "ImageNet1k": "class_imagenet1k",
    "INaturalist": "class_inaturalist",
    "InternVid": "caption_internvid",
    "Kinetics": "class_kinetics",
    "LAION400M": "caption_laion400m",
    "LAION5B": "caption_laion5b",
    "LSUN": "class_lsun",
    "LVIS": "detect_lvis",
    "MSCOCO": "detect_mscoco",
    "MSCOCOCaption": "caption_mscoco",
    "MVTecAD": "class_mvtec_ad",
    "OpenImagesV4": "class_openimages_v4",
    "OpenVid1M": "caption_openvid1m",
    "Places": "class_places",
    "TextCaps": "caption_textcaps",
    "UCF101": "class_ucf101",
    "VisualCoT": "qa_visual_cot",
    "VisualGenome": "qa_visual_genome",
    "VQAv2": "qa_vqa_v2",
}


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
