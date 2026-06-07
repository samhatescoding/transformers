from __future__ import annotations

from .hf_common import HFClassificationDataset


class Cityscapes(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "scene_parse_150") -> None:
        super().__init__(
            name="cityscapes",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image", "leftImg8bit", "rgb"),
            label_keys=("label", "semantic_label", "class"),
            fallback_labels=[
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
            ],
        )
