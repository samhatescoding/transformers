from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class LAION400M(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "laion/laion400m") -> None:
        super().__init__(
            name="laion400m",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            answer_keys=("caption", "text", "TEXT"),
        )
