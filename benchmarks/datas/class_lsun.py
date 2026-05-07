from __future__ import annotations

from .hf_common import HFClassificationDataset


class LSUN(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "pcuenq/lsun-bedrooms") -> None:
        super().__init__(name="lsun", dataset_id=dataset_id, split=split, streaming=streaming, fallback_labels=("bedroom",))

    def get_labels_img(self, row):
        del row
        return ["bedroom"]
