from __future__ import annotations

import ast
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent / "type_l"
CACHE = Path(__file__).resolve().parents[1] / ".tmp" / "type_l_metadata"

OPENIMAGES_URL = (
    "https://storage.googleapis.com/openimages/2018_04/"
    "class-descriptions-boxable.csv"
)
PLACES_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/"
    "categories_places365.txt"
)
INATURALIST_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/"
    "train_val2017.zip"
)

STATIC_LABELS = {
    "cityscapes.txt": [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ],
    "fairface.txt": [
        "0-2",
        "3-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "more than 70",
    ],
    "fashion_mnist.txt": [
        "T - shirt / top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "lsun.txt": [
        "bedroom",
        "bridge",
        "church outdoor",
        "classroom",
        "conference room",
        "dining room",
        "kitchen",
        "living room",
        "restaurant",
        "tower",
    ],
    "mvtec_ad.txt": ["normal", "defective"],
    "dfdc.txt": ["real", "fake"],
}


def _download(url: str, filename: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if not path.exists():
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "transformers-benchmark-label-generator/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def _write(filename: str, labels: list[str], expected_count: int) -> None:
    cleaned = [str(label).strip() for label in labels if str(label).strip()]
    if len(cleaned) != expected_count:
        raise ValueError(
            f"{filename} expected {expected_count} labels, found {len(cleaned)}."
        )
    if len({label.casefold() for label in cleaned}) != len(cleaned):
        raise ValueError(f"{filename} contains duplicate labels.")
    (ROOT / filename).write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _imagenet_labels() -> list[str]:
    path = Path(
        hf_hub_download(
            "ILSVRC/imagenet-1k",
            "classes.py",
            repo_type="dataset",
        )
    )
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "IMAGENET2012_CLASSES"
            for target in node.targets
        ):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not call.args:
            continue
        mapping = ast.literal_eval(call.args[0])
        return list(mapping.values())
    raise ValueError("Could not parse IMAGENET2012_CLASSES.")


def _inaturalist_species() -> list[str]:
    archive = _download(INATURALIST_URL, "train_val2017.zip")
    with zipfile.ZipFile(archive) as zipped:
        payload = json.loads(zipped.read("train2017.json"))
    categories = sorted(payload["categories"], key=lambda category: category["id"])
    # iNaturalist 2017 also contains genus, subspecies, varieties, and hybrids.
    return [
        category["name"]
        for category in categories
        if len(str(category["name"]).split()) == 2
    ]


def _openimages_boxable_labels() -> list[str]:
    path = _download(
        OPENIMAGES_URL,
        "class-descriptions-boxable.csv",
    )
    with path.open(encoding="utf-8", newline="") as file:
        return [row[1] for row in csv.reader(file) if len(row) >= 2]


def _places_labels() -> list[str]:
    path = _download(PLACES_URL, "categories_places365.txt")
    indexed_labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw_name, raw_index = line.strip().rsplit(" ", 1)
        label = raw_name.split("/", 2)[-1].replace("_", " ").replace("/", " ")
        indexed_labels[int(raw_index)] = label.strip()
    return [indexed_labels[index] for index in range(365)]


def _kinetics_labels() -> list[str]:
    path = Path(
        hf_hub_download(
            "iejMac/CLIP-Kinetics700",
            "data/annotations/deepmind/kinetics700_2020/train.csv",
            repo_type="dataset",
        )
    )
    with path.open(encoding="utf-8", newline="") as file:
        return sorted({row["label"].strip() for row in csv.DictReader(file)})


def _ucf101_labels() -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset("flwrlabs/ucf101", split="train", streaming=True)
    return list(dataset.features["label"].names)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for filename, labels in STATIC_LABELS.items():
        _write(filename, labels, len(labels))

    _write("imagenet1k.txt", _imagenet_labels(), 1000)
    _write("inaturalist2017.txt", _inaturalist_species(), 4895)
    _write("openimages_v4.txt", _openimages_boxable_labels(), 601)
    _write("places365.txt", _places_labels(), 365)
    _write("kinetics700.txt", _kinetics_labels(), 700)
    _write("ucf101.txt", _ucf101_labels(), 101)

    for path in sorted(ROOT.glob("*.txt")):
        count = len(path.read_text(encoding="utf-8").splitlines())
        print(f"{path.name}: {count}")


if __name__ == "__main__":
    main()
