from __future__ import annotations

from .hf_common import HFCaptionDataset


class LAION5B(HFCaptionDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "nousr/laion5b-subset-and-cliph-embeddings",
    ) -> None:
        super().__init__(
            name="laion5b",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("jpg",),
            caption_keys=("caption", "__key__"),
        )
