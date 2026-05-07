from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from .hf_common import HFClassificationDataset


class Places(HFClassificationDataset):
    LABELS_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"

    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "dpdl-benchmark/Places365-Validation") -> None:
        actual_split = "train" if split.startswith("val") else split
        super().__init__(name="places", dataset_id=dataset_id, split=actual_split, streaming=streaming)
        if self.labels and all(str(label).isdigit() for label in self.labels):
            resolved_labels = self._load_places365_labels()
            if resolved_labels:
                self.labels = resolved_labels

    def _load_places365_labels(self) -> list[str]:
        cache_path = Path(".tmp") / "places365" / "categories_places365.txt"
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(self.LABELS_URL, timeout=60) as response:
                cache_path.write_bytes(response.read())

        labels = [""] * 365
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            raw_name, raw_index = stripped.rsplit(" ", 1)
            index = int(raw_index)
            label = raw_name.split("/", 2)[-1].replace("_", " ").replace("/", " ")
            if 0 <= index < len(labels):
                labels[index] = label.strip()

        return [label for label in labels if label]
