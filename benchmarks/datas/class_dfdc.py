from __future__ import annotations

from .hf_common import HFVideoClassificationDataset


class DFDC(HFVideoClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "mkhLlamaLearn/dfdcpics2") -> None:
        super().__init__(
            name="dfdc",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            frame_keys=("image",),
            label_keys=("label",),
            fallback_labels=("real", "fake"),
        )
