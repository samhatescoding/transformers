from __future__ import annotations

from .hf_common import HFClassificationDataset


class Kinetics(HFClassificationDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "iejMac/CLIP-Kinetics700",
    ) -> None:
        super().__init__(
            name="kinetics",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            label_keys=("label",),
        )

    def _standardize_row(self, row):
        out = dict(row)
        out["image"] = self._youtube_thumbnail_url(row.get("youtube_id"))
        return out
