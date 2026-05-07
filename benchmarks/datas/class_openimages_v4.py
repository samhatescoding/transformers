from __future__ import annotations

from .hf_common import HFClassificationDataset


class OpenImagesV4(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "vikhyatk/openimages-bbox") -> None:
        super().__init__(
            name="openimages_v4",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            label_keys=("label", "labels", "class"),
        )

    def get_labels_img(self, row):
        labels = []
        for obj in row.get("objects", []):
            label = str(obj.get("label", "")).strip()
            if label:
                labels.append(label)
        return labels

    def get_annotations_for_row(self, row):
        annotations = []
        for obj in row.get("objects", []):
            label = str(obj.get("label", "")).strip()
            xmin = obj.get("xmin")
            ymin = obj.get("ymin")
            xmax = obj.get("xmax")
            ymax = obj.get("ymax")
            if not label:
                continue
            if not all(isinstance(value, (int, float)) for value in (xmin, ymin, xmax, ymax)):
                continue
            annotations.append(
                {
                    "label": label,
                    "bbox": [float(xmin), float(ymin), float(xmax) - float(xmin), float(ymax) - float(ymin)],
                }
            )
        return annotations
