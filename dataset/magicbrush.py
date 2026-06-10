from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class MagicBrush(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "osunlp/MagicBrush") -> None:
        super().__init__(
            name="magicbrush",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="pair",
            source_image_keys=("source_image", "source_img", "input_image", "image", "src_img"),
            target_image_keys=("target_image", "target_img", "edited_image", "output_image", "tgt_img"),
            question_keys=("question", "instruction", "edit_instruction"),
            answer_keys=("answer", "instruction", "edit_instruction"),
        )
