from __future__ import annotations

from .hf_common import HFCaptionDataset


class TextCaps(HFCaptionDataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "lmms-lab/TextCaps") -> None:
        actual_split = "val" if split.startswith("val") else split
        super().__init__(
            name="textcaps",
            dataset_id=dataset_id,
            split=actual_split,
            streaming=streaming,
            caption_keys=("reference_strs", "caption_str", "captions", "caption"),
        )
