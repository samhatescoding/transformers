"""Benchmark lookup for generic fine-tuning and evaluation scripts."""

from __future__ import annotations

from benchmarks import (
    BLIP3o60kBenchmark,
    ConceptualCaptionsBenchmark,
    ConceptualCaptionsCaptionBenchmark,
    DFDCBenchmark,
    DocVQABenchmark,
    FairFaceBenchmark,
    FashionMNISTBenchmark,
    Flickr30kBenchmark,
    Flickr30kEntitiesBenchmark,
    GQABenchmark,
    ImageNet1kBenchmark,
    INaturalistBenchmark,
    InternVidBenchmark,
    KineticsBenchmark,
    LAION400MBenchmark,
    LAION5BBenchmark,
    LSUNBenchmark,
    LVISBenchmark,
    MSCOCOBenchmark,
    MSCOCOCaptionBenchmark,
    MVTecADBenchmark,
    OpenImagesV4Benchmark,
    OpenImagesV4DetectionBenchmark,
    OpenVid1MBenchmark,
    PlacesBenchmark,
    TextCapsBenchmark,
    UCF101Benchmark,
    VisualCoTBenchmark,
    VisualGenomeBenchmark,
    VQAv2Benchmark,
)


BENCHMARK_CLASSES = {
    benchmark_cls.benchmark_name: benchmark_cls
    for benchmark_cls in (
        BLIP3o60kBenchmark,
        ConceptualCaptionsBenchmark,
        ConceptualCaptionsCaptionBenchmark,
        DFDCBenchmark,
        DocVQABenchmark,
        FairFaceBenchmark,
        FashionMNISTBenchmark,
        Flickr30kBenchmark,
        Flickr30kEntitiesBenchmark,
        GQABenchmark,
        ImageNet1kBenchmark,
        INaturalistBenchmark,
        InternVidBenchmark,
        KineticsBenchmark,
        LAION400MBenchmark,
        LAION5BBenchmark,
        LSUNBenchmark,
        LVISBenchmark,
        MSCOCOBenchmark,
        MSCOCOCaptionBenchmark,
        MVTecADBenchmark,
        OpenImagesV4Benchmark,
        OpenImagesV4DetectionBenchmark,
        OpenVid1MBenchmark,
        PlacesBenchmark,
        TextCapsBenchmark,
        UCF101Benchmark,
        VisualCoTBenchmark,
        VisualGenomeBenchmark,
        VQAv2Benchmark,
    )
}

