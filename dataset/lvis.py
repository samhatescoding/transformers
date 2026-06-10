from __future__ import annotations

from typing import Any, Dict, List

from .hf_common import HFBaseDataset


class LVIS(HFBaseDataset):
    def __init__(
        self,
        split: str = "validation",
        streaming: bool = True,
        dataset_id: str = "fw407/lvis",
    ) -> None:
        if split == "validation":
            split = "train"
        super().__init__(
            name="lvis",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
        )
        self.category_id_to_label = self._extract_category_map()
        self.labels = list(self.category_id_to_label.values())

    def get_image_from_row(self, row):
        return self._extract_image(row, ("image",))

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        objects = row.get("objects") or {}
        boxes = objects.get("bboxes", objects.get("bbox", [])) if isinstance(objects, dict) else []
        classes = objects.get("classes", objects.get("category_id", [])) if isinstance(objects, dict) else []
        annotations: List[Dict[str, Any]] = []
        for raw_box, raw_class in zip(boxes, classes):
            box = raw_box[0] if isinstance(raw_box, list) and len(raw_box) == 1 and isinstance(raw_box[0], list) else raw_box
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            label = self._label_for_class(raw_class)
            annotations.append({"bbox": [float(value) for value in box], "label": label})
        return annotations

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        return [item["label"] for item in self.get_annotations_for_row(row) if item["label"]]

    def get_labels(self, rows) -> List[str]:
        labels: List[str] = []
        seen = set()
        for row in rows:
            for label in self.get_labels_img(row):
                normalized = self.normalize_text(label)
                if normalized not in seen:
                    seen.add(normalized)
                    labels.append(label)
        self.labels = labels
        return labels

    def _label_for_class(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        try:
            class_id = int(value)
        except (TypeError, ValueError):
            return ""
        return self.category_id_to_label.get(class_id, str(class_id))

    def _extract_category_map(self) -> Dict[int, str]:
        try:
            objects = self.ds.features["objects"]
            classes = objects["classes"] if isinstance(objects, dict) else objects.feature["classes"]
            feature = getattr(classes, "feature", classes)
            names = list(getattr(feature, "names", []))
            return {index: str(name) for index, name in enumerate(names)}
        except Exception:
            return {}
