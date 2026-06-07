from __future__ import annotations

from .hf_common import HFMultipleChoiceSourceDataset


class HDTF(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "HDTF/HDTF") -> None:
        super().__init__(
            name="hdtf",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="video",
            frame_keys=("frames", "video_frames", "images", "image"),
            question_keys=("question", "prompt", "text", "transcript"),
            answer_keys=("answer", "caption", "text", "transcript"),
        )
