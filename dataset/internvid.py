from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class InternVid(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "OpenGVLab/InternVid") -> None:
        super().__init__(
            name="internvid",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="video",
            answer_keys=("caption", "text", "answer"),
        )
