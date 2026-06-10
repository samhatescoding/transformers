from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class ImgEdit(HFMultipleChoiceSourceDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "diffusion-cot/imgedit-simpler",
    ) -> None:
        super().__init__(
            name="imgedit",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="pair",
            source_image_keys=("original_image",),
            target_image_keys=("edited_image",),
            question_keys=("question", "instruction", "edit_instruction", "prompt"),
            answer_keys=("answer", "instruction", "edit_instruction", "prompt"),
        )
