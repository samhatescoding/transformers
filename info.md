# Benchmark Structure Review

## Current benchmark tree structure

The benchmark tree is now centralized in one canonical place: `benchmarks/datas/benchmark_wrappers.py`.

Public task-level parents are exposed from `benchmarks/__init__.py`, and dataset files under `benchmarks/datas/` now contain dataset loaders only.

```text
BaseBenchmark
+-- CaptioningBenchmark
|   +-- Flickr30kBenchmark
|   +-- MSCOCOCaptionBenchmark
|   `-- TextCapsBenchmark
+-- ClassificationBenchmark
|   +-- FairFaceBenchmark
|   +-- FashionMNISTBenchmark
|   +-- ImageNet1kBenchmark
|   +-- INaturalistBenchmark
|   +-- LSUNBenchmark
|   +-- MVTecADBenchmark
|   +-- OpenImagesV4Benchmark
|   `-- PlacesBenchmark
+-- DetectionBenchmark
|   +-- Flickr30kEntitiesBenchmark
|   +-- MSCOCOBenchmark
|   |   `-- LVISBenchmark
|   `-- OpenImagesV4DetectionBenchmark
+-- MultipleChoiceBenchmark
|   +-- BLIP3o60kBenchmark
|   +-- ConceptualCaptionsBenchmark
|   +-- InternVidBenchmark
|   +-- LAION400MBenchmark
|   +-- LAION5BBenchmark
|   `-- OpenVid1MBenchmark
+-- VideoClassificationBenchmark
|   +-- DFDCBenchmark
|   +-- KineticsBenchmark
|   `-- UCF101Benchmark
+-- VisualQABenchmark
|   +-- DocVQABenchmark
|   +-- GQABenchmark
|   +-- VisualCoTBenchmark
|   +-- VisualGenomeBenchmark
|   `-- VQAv2Benchmark
```

## Why this is better

- There is now one source of truth for benchmark classes.
- Dataset modules no longer mix dataset-loading concerns with benchmark-definition concerns.
- The public package exports match the real tree, including `MultipleChoiceBenchmark` and `VideoClassificationBenchmark`.
- `UCF101Benchmark` is now part of the video-classification branch instead of being a one-off direct child of `BaseBenchmark`.
- The structure is easier to scan, reason about, and extend without creating parallel inheritance paths.

## Rating

I would now rate it **10/10** for this repo's current scope.

Why:

- The hierarchy is consistent: base task classes first, concrete dataset benchmarks second.
- Responsibilities are separated cleanly.
- Public imports reflect the actual architecture.
- There is no longer a duplicate benchmark tree hidden inside dataset files.

The only meaningful next step beyond this would be naming cleanup, such as renaming `benchmark_wrappers.py` to something more explicit like `dataset_benchmarks.py`, but that is cosmetic rather than structural.
