from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class HQEdit(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "UCSC-VLAA/HQ-Edit") -> None:
        super().__init__(
            name="hq_edit",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="pair",
            source_image_keys=("source_image", "input_image", "image", "src_img"),
            target_image_keys=("target_image", "edited_image", "output_image", "tgt_img"),
            question_keys=("question", "instruction", "edit_instruction", "prompt"),
            answer_keys=("answer", "instruction", "edit_instruction", "prompt"),
        )
