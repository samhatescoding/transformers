from __future__ import annotations

from .hf_common import HFCaptionDataset


class LAION400M(HFCaptionDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "tempertrash/laion_400m",
    ) -> None:
        super().__init__(
            name="laion400m",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            caption_keys=("caption", "captions", "text", "TEXT"),
        )
