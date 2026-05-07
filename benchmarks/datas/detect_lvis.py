from __future__ import annotations

from .hf_common import HFClassificationDataset


class LVIS(HFClassificationDataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "lvis/lvis") -> None:
        super().__init__(
            name="lvis",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            label_keys=("label", "labels", "category", "categories"),
        )
