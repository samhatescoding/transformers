from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class BLIP3o60k(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "blip3o/blip3o-60k") -> None:
        super().__init__(
            name="blip3o_60k",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="pair",
            source_image_keys=("source_image", "source_img", "image_before"),
            target_image_keys=("target_image", "target_img", "image_after"),
            answer_keys=("instruction", "prompt", "answer"),
        )
