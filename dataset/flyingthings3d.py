from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class FlyingThings3D(HFMultipleChoiceSourceDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "infinity1096/flyingthings3d_processed",
    ) -> None:
        super().__init__(
            name="flyingthings3d",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="image",
            image_keys=("image", "left_image", "frame", "rgb"),
            question_keys=("question", "prompt", "description"),
            answer_keys=("answer", "caption", "text", "description"),
        )
