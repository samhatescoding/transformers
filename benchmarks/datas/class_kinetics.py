from __future__ import annotations

from .hf_common import HFVideoClassificationDataset


class Kinetics(HFVideoClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "google/kinetics700") -> None:
        super().__init__(name="kinetics", dataset_id=dataset_id, split=split, streaming=streaming)
