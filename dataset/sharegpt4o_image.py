from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class ShareGPT4oImage(HFMultipleChoiceSourceDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "FreedomIntelligence/ShareGPT-4o-Image",
    ) -> None:
        super().__init__(
            name="sharegpt4o_image",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="pair",
            source_image_keys=("source_image", "input_image", "image", "src_img"),
            target_image_keys=("target_image", "target_image", "edited_image", "output_image", "tgt_img"),
            question_keys=("question", "instruction", "prompt", "caption"),
            answer_keys=("answer", "instruction", "prompt", "caption"),
        )
