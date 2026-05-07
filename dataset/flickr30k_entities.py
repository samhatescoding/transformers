from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from PIL import Image

from ._base_dataset import BaseDataset


class Flickr30kEntities(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "Rajarshi-Roy-research/Flickr30k_Grounding_Som",
    ) -> None:
        self.name = "flickr30k_entities"
        self.split = split
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.labels: List[str] = []
        self.ds = load_dataset(dataset_id, split=split, streaming=streaming)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for row in self.ds:
            yield self._standardize_row(row)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for index, row in enumerate(self.ds):
            if index >= n:
                break
            samples.append(self._standardize_row(row))
        return samples

    def get_labels(self, rows) -> List[str]:
        merged: List[str] = []
        seen = set()
        for row in rows:
            for label in self.get_labels_img(row):
                normalized = self.normalize_text(label)
                if normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(label)
        self.labels = merged
        return merged

    def get_labels_img(self, row) -> List[str]:
        return [item["label"] for item in self.get_annotations_for_row(row)]

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        boxes = self._parse_boxes(row.get("json_data"))
        labels = self._parse_phrase_labels(str(row.get("caption", "")))
        annotations: List[Dict[str, Any]] = []
        for label, xyxy in zip(labels, boxes):
            annotations.append({"label": label, "bbox": self._to_xywh(xyxy)})
        return annotations

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError("Flickr30k Entities row is missing a decoded image.")

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["annotations"] = self.get_annotations_for_row(row)
        return out

    def _parse_phrase_labels(self, caption: str) -> List[str]:
        labels = [match.strip() for match in re.findall(r"\(([^)]+)\)", caption) if match.strip()]
        if labels:
            return labels
        fallback = str(caption).strip()
        return [fallback] if fallback else []

    def _parse_boxes(self, raw: Any) -> List[List[float]]:
        text = str(raw or "")
        matches = re.findall(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            text,
        )
        if not matches:
            return []
        boxes: List[List[float]] = []
        for x0, y0, x1, y1 in matches:
            boxes.append([float(x0), float(y0), float(x1), float(y1)])
        return boxes

    def _to_xywh(self, xyxy: List[float]) -> List[float]:
        x0, y0, x1, y1 = xyxy
        return [x0, y0, x1 - x0, y1 - y0]
