from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from ._base_dataset import BaseDataset


class INaturalistDetection(BaseDataset):
    """iNaturalist images with manually annotated species-instance boxes."""

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "philipp-zettl/inaturalist-bbs",
    ) -> None:
        del split
        self.name = "inaturalist_detection"
        self.split = "train"
        self.streaming = streaming
        self.dataset_id = dataset_id
        files = HfApi().list_repo_files(repo_id=dataset_id, repo_type="dataset")
        self.annotation_files = sorted(
            (path for path in files if path.startswith("annotations/") and path.endswith(".json")),
            key=lambda path: int(path.rsplit("/", 1)[-1].removesuffix(".json")),
        )
        self.labels: List[str] = []
        self._row_cache: Dict[str, Dict[str, Any]] = {}

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for annotation_file in self.annotation_files:
            yield self._load_row(annotation_file)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return [self._load_row(path) for path in self.annotation_files[:n]]

    def get_labels(self, rows) -> List[str]:
        labels: List[str] = []
        seen = set()
        for row in rows:
            label = str(row.get("species_name", "")).strip()
            normalized = self.normalize_text(label)
            if label and normalized not in seen:
                seen.add(normalized)
                labels.append(label)
        self.labels = labels
        return labels

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        label = str(row.get("species_name", "")).strip()
        return [label] if label else []

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        label = str(row.get("species_name", "")).strip()
        annotations: List[Dict[str, Any]] = []
        for box in row.get("bounding_boxes", []):
            if not isinstance(box, dict):
                continue
            values = [box.get(key) for key in ("x", "y", "width", "height")]
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            annotations.append({"label": label, "bbox": [float(value) for value in values]})
        return annotations

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError("iNaturalist detection row is missing its decoded image.")

    def _load_row(self, annotation_file: str) -> Dict[str, Any]:
        cached = self._row_cache.get(annotation_file)
        if cached is not None:
            return dict(cached)

        path = hf_hub_download(
            repo_id=self.dataset_id,
            filename=annotation_file,
            repo_type="dataset",
        )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        original_data = payload.get("original_data", {})
        encoded_image = original_data.get("image") if isinstance(original_data, dict) else None
        if not encoded_image:
            raise ValueError(f"{annotation_file} does not contain an embedded image.")
        image = Image.open(BytesIO(base64.b64decode(encoded_image))).convert("RGB")
        row = {
            "id": payload.get("image_index"),
            "image": image,
            "species_name": str(payload.get("species_name", "")).strip(),
            "bounding_boxes": list(payload.get("bounding_boxes", [])),
        }
        self._row_cache[annotation_file] = row
        return dict(row)
