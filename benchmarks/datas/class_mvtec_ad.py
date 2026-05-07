from __future__ import annotations

from .hf_common import HFClassificationDataset


class MVTecAD(HFClassificationDataset):
    def __init__(self, split: str = "test", streaming: bool = True, dataset_id: str = "katiehahm/mvtec_ad") -> None:
        super().__init__(
            name="mvtec_ad",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            label_keys=("gt_label",),
            fallback_labels=("normal", "defective"),
        )

    def get_labels_img(self, row):
        raw = row.get("gt_label")
        if raw is None:
            return []
        return ["defective" if int(raw) else "normal"]
