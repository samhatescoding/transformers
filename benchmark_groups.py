from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


BENCHMARK_TYPE_TITLES = {
    "A": "Answering Questions",
    "B": "Bounding Box Detection",
    "C": "Captioning",
    "E": "Editing Reconstruction",
    "G": "Generating Reconstruction",
    "L": "Labeling",
    "P": "Preference",
    "R": "Rating",
}


ALL_BENCHMARKS_BY_TYPE = {
    "A": [
        {"name": "FlyingThings3D", "module": "benchmarks.multiple_choice.flyingthings3d", "class": "FlyingThings3DBenchmark"},
        {"name": "Visual Genome", "module": "benchmarks.visual_qa.visual_genome", "class": "VisualGenomeBenchmark"},
        {"name": "VQA v2.0", "module": "benchmarks.visual_qa.vqa_v2", "class": "VQAv2Benchmark"},
        {"name": "GQA", "module": "benchmarks.visual_qa.gqa", "class": "GQABenchmark"},
        {"name": "DocVQA", "module": "benchmarks.visual_qa.docvqa", "class": "DocVQABenchmark"},
        {"name": "Visual CoT", "module": "benchmarks.visual_qa.visual_cot", "class": "VisualCoTBenchmark"},
    ],
    "B": [
        {"name": "Flickr30k Entities", "module": "benchmarks.detection.flickr30k_entities", "class": "Flickr30kEntitiesBenchmark"},
        {"name": "MS COCO", "module": "benchmarks.detection.mscoco", "class": "MSCOCOBenchmark"},
        {"name": "LVIS", "module": "benchmarks.detection.lvis", "class": "LVISBenchmark"},
        {"name": "OpenImages V4", "module": "benchmarks.detection.openimages_v4_detection", "class": "OpenImagesV4DetectionBenchmark"},
        {"name": "iNaturalist", "module": "benchmarks.detection.inaturalist", "class": "INaturalistDetectionBenchmark"},
        {"name": "Visual CoT", "module": "benchmarks.detection.visual_cot", "class": "VisualCoTDetectionBenchmark"},
    ],
    "C": [
        {"name": "Flickr30k", "module": "benchmarks.captioning.flickr30k", "class": "Flickr30kBenchmark"},
        {"name": "MS COCO Captions", "module": "benchmarks.captioning.mscoco_caption", "class": "MSCOCOCaptionBenchmark"},
        {"name": "TextCaps", "module": "benchmarks.captioning.textcaps", "class": "TextCapsBenchmark"},
        {"name": "Conceptual Captions", "module": "benchmarks.captioning.conceptual_captions_caption", "class": "ConceptualCaptionsCaptionBenchmark"},
        {"name": "HDTF", "module": "benchmarks.captioning.hdtf", "class": "HDTFBenchmark"},
        {"name": "InternVid", "module": "benchmarks.captioning.internvid", "class": "InternVidBenchmark"},
        {"name": "LAION-400M", "module": "benchmarks.captioning.laion400m", "class": "LAION400MBenchmark"},
        {"name": "LAION-5B", "module": "benchmarks.captioning.laion5b", "class": "LAION5BBenchmark"},
        {"name": "OpenVid-1M", "module": "benchmarks.captioning.openvid1m", "class": "OpenVid1MCaptionBenchmark"},
    ],
    "E": [
        {"name": "HQ-Edit", "module": "benchmarks.multiple_choice.hq_edit", "class": "HQEditBenchmark"},
        {"name": "ImgEdit", "module": "benchmarks.multiple_choice.imgedit", "class": "ImgEditBenchmark"},
        {"name": "MagicBrush", "module": "benchmarks.multiple_choice.magicbrush", "class": "MagicBrushBenchmark"},
        {"name": "ShareGPT-4o-Image", "module": "benchmarks.multiple_choice.sharegpt4o_image_edit", "class": "ShareGPT4oImageEditBenchmark"},
    ],
    "G": [
        {"name": "BLIP3o-60k", "module": "benchmarks.multiple_choice.blip3o_60k", "class": "BLIP3o60kBenchmark"},
        {"name": "Conceptual Captions", "module": "benchmarks.multiple_choice.conceptual_captions", "class": "ConceptualCaptionsBenchmark"},
        {"name": "DiffusionDB", "module": "benchmarks.multiple_choice.diffusiondb", "class": "DiffusionDBBenchmark"},
        {"name": "OpenVid-1M", "module": "benchmarks.multiple_choice.openvid1m", "class": "OpenVid1MBenchmark"},
        {"name": "ShareGPT4o-Image", "module": "benchmarks.multiple_choice.sharegpt4o_image", "class": "ShareGPT4oImageBenchmark"},
    ],
    "L": [
        {"name": "Cityscapes", "module": "benchmarks.classification.cityscapes", "class": "CityscapesBenchmark"},
        {"name": "FairFace", "module": "benchmarks.classification.fairface", "class": "FairFaceBenchmark"},
        {"name": "Fashion-MNIST", "module": "benchmarks.classification.fashion_mnist", "class": "FashionMNISTBenchmark"},
        {"name": "ImageNet-1K", "module": "benchmarks.classification.imagenet1k", "class": "ImageNet1kBenchmark"},
        {"name": "iNaturalist", "module": "benchmarks.classification.inaturalist", "class": "INaturalistBenchmark"},
        {"name": "LSUN", "module": "benchmarks.classification.lsun", "class": "LSUNBenchmark"},
        {"name": "MVTec AD", "module": "benchmarks.classification.mvtec_ad", "class": "MVTecADBenchmark"},
        {"name": "OpenImages V4", "module": "benchmarks.classification.openimages_v4", "class": "OpenImagesV4Benchmark"},
        {"name": "Places", "module": "benchmarks.classification.places", "class": "PlacesBenchmark"},
        {"name": "DFDC", "module": "benchmarks.video_classification.dfdc", "class": "DFDCBenchmark"},
        {"name": "Kinetics", "module": "benchmarks.video_classification.kinetics", "class": "KineticsBenchmark"},
        {"name": "UCF101", "module": "benchmarks.video_classification.ucf101", "class": "UCF101Benchmark"},
    ],
    "P": [
        {"name": "Pick-a-Pic", "module": "benchmarks.multiple_choice.pick_a_pic", "class": "PickAPicBenchmark"},
    ],
    "R": [
        {"name": "TAD66K", "module": "benchmarks.classification.tad66k", "class": "TAD66KBenchmark"},
    ],
}


BenchmarkSelection = Mapping[str, str | Sequence[str]]


def selected_benchmark_specs(selection_by_type: BenchmarkSelection) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for type_code, selection in selection_by_type.items():
        available = ALL_BENCHMARKS_BY_TYPE[type_code]
        if selection == "ALL":
            names = {item["name"] for item in available}
        else:
            names = set(selection)
            unknown = names - {item["name"] for item in available}
            if unknown:
                raise ValueError(f"Unknown benchmarks for type {type_code}: {sorted(unknown)}")
        for item in available:
            if item["name"] in names:
                selected.append({**item, "type": type_code, "type_title": BENCHMARK_TYPE_TITLES[type_code]})
    return selected


def all_benchmark_specs() -> list[dict[str, Any]]:
    return selected_benchmark_specs({type_code: "ALL" for type_code in ALL_BENCHMARKS_BY_TYPE})
