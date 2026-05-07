from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class OpenVid1M(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "OpenGVLab/OpenVid-1M") -> None:
        super().__init__(
            name="openvid1m",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="video",
            answer_keys=("prompt", "caption", "text"),
        )
