from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .hf_common import HFClassificationDataset


class Cityscapes(HFClassificationDataset):
    LABELS = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "tanganke/cityscapes") -> None:
        super().__init__(
            name="cityscapes",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            label_keys=("segmentation_19", "segmentation"),
            fallback_labels=self.LABELS,
        )
        self.labels = list(self.LABELS)

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        value = row.get("image")
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.dtype != np.uint8:
            scale = 255.0 if float(array.max(initial=0)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        value = row.get("segmentation_19", row.get("segmentation"))
        if value is None:
            return []
        ids = sorted(int(item) for item in np.unique(np.asarray(value)) if 0 <= int(item) < len(self.LABELS))
        return [self.LABELS[index] for index in ids]
