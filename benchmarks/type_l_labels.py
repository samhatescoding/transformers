from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


LABELS_ROOT = (
    Path(__file__).resolve().parents[1] / "benchmark_choices" / "type_l"
)

LABEL_FILES = {
    "cityscapes": "cityscapes.txt",
    "fairface": "fairface.txt",
    "fashion_mnist": "fashion_mnist.txt",
    "imagenet-1k": "imagenet1k.txt",
    "imagenet1k": "imagenet1k.txt",
    "inaturalist": "inaturalist2017.txt",
    "lsun": "lsun.txt",
    "mvtec_ad": "mvtec_ad.txt",
    "openimages_v4": "openimages_v4.txt",
    "places": "places365.txt",
    "dfdc": "dfdc.txt",
    "kinetics": "kinetics700.txt",
    "ucf101": "ucf101.txt",
}


@lru_cache(maxsize=None)
def _load_label_file(filename: str) -> tuple[str, ...]:
    path = LABELS_ROOT / filename
    labels = tuple(path.read_text(encoding="utf-8").splitlines())
    if not labels or any(not label.strip() for label in labels):
        raise ValueError(f"{path} must contain one nonblank label per line.")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError(f"{path} contains duplicate labels.")
    return labels


def get_complete_type_l_labels(benchmark: Any) -> list[str] | None:
    candidates = (
        getattr(benchmark, "benchmark_name", None),
        getattr(benchmark, "name", None),
        getattr(getattr(benchmark, "dataset", None), "name", None),
    )
    for candidate in candidates:
        filename = LABEL_FILES.get(str(candidate or "").strip().casefold())
        if filename is not None:
            return list(_load_label_file(filename))
    return None
