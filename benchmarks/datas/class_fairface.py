from __future__ import annotations

from .hf_common import HFClassificationDataset


class FairFace(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "HuggingFaceM4/FairFace") -> None:
        super().__init__(
            name="fairface",
            dataset_id=dataset_id,
            config_name="0.25",
            split=split,
            streaming=streaming,
            label_keys=("age",),
        )
