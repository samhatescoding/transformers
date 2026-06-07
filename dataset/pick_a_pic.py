from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class PickAPic(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "kevinkingslin/pick-a-pic") -> None:
        super().__init__(
            name="pick_a_pic",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image", "jpg_0", "image_0", "chosen_image"),
            question_keys=("question", "prompt", "caption"),
            answer_keys=("answer", "prompt", "caption"),
        )
