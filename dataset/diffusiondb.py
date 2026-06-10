from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class DiffusionDB(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "poloclub/diffusiondb") -> None:
        super().__init__(
            name="diffusiondb",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            revision="refs/convert/parquet",
            image_keys=("image",),
            question_keys=("question", "prompt"),
            answer_keys=("answer", "prompt", "text"),
        )
