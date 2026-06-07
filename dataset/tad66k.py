from __future__ import annotations

from .hf_common import HFClassificationDataset


class TAD66K(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "tad66k") -> None:
        super().__init__(
            name="tad66k",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            label_keys=("theme", "label", "category", "class"),
        )
