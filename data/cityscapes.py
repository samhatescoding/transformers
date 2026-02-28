from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
from datasets import load_dataset

from .base import BaseDataset


class Cityscapes(BaseDataset):
    """
    Cityscapes-style dataset adapter for image workflows.

    Backward-compatible alias `Citiscapes` is provided below.
    """

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "huggan/cityscapes",
        use_auth_token: Optional[bool] = None,
    ):
        self.name = "cityscapes"
        self.split = split
        self.streaming = streaming

        self.ds = load_dataset(
            dataset_id,
            split=split,
            streaming=streaming,
            use_auth_token=use_auth_token,
        )

        self.class_labels: List[str] = self._extract_class_labels()
        if not self.class_labels:
            # Standard 19 Cityscapes semantic classes
            self.class_labels = [
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
        self.labels = list(self.class_labels)

    def __repr__(self) -> str:
        return f"<Dataset {self.name} | split={self.split} | streaming={self.streaming}>"

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.ds)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(self.ds):
            if i >= n:
                break
            samples.append(row)
        return samples

    def _extract_class_labels(self) -> List[str]:
        try:
            feature_key = "label" if "label" in self.ds.features else "class"
            feature = self.ds.features[feature_key]
            names = list(getattr(feature, "names", []))
            if names:
                return names
        except Exception:
            pass
        return []

    def _label_from_row(self, row: Dict[str, Any]) -> Optional[str]:
        raw = row.get("label", row.get("class"))
        if raw is None:
            return None
        if isinstance(raw, int) and 0 <= raw < len(self.class_labels):
            return self.class_labels[raw]
        return str(raw)

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        label = self._label_from_row(row)
        return [label] if label else list(self.class_labels)

    def get_labels(self, rows) -> List[str]:
        merged: List[str] = []
        seen = set()
        for row in rows:
            for label in self.get_labels_img(row):
                nlabel = self.normalize_text(label)
                if nlabel in seen:
                    continue
                seen.add(nlabel)
                merged.append(label)
        self.labels = merged
        return merged

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image_key_order = (
            "image",
            "leftImg8bit",
            "left_image",
            "rgb",
            "img",
            "pixel_values",
            "image_left",
            "image_rgb",
        )
        attempted: List[str] = []
        for key in image_key_order:
            if key in row and row[key] is not None:
                try:
                    return self._image_from_any(row[key]).convert("RGB")
                except Exception:
                    attempted.append(key)

        # Fallback: search recursively for any decodable image-like payload.
        for key, value in self._iter_nested_items(row):
            if value is None:
                continue
            try:
                return self._image_from_any(value).convert("RGB")
            except Exception:
                attempted.append(key)

        raise ValueError(
            "Could not extract an RGB image from this row. "
            f"Row keys: {sorted(row.keys())}. "
            f"Attempted fields: {sorted(set(attempted))}."
        )

    @staticmethod
    def _image_from_any(obj: Any) -> Image.Image:
        if isinstance(obj, Image.Image):
            return obj

        # Hugging Face image objects often expose `.convert`.
        if hasattr(obj, "convert"):
            try:
                return obj.convert("RGB")
            except Exception:
                pass

        if isinstance(obj, dict):
            if "bytes" in obj and obj["bytes"] is not None:
                return Image.open(BytesIO(obj["bytes"]))
            if "path" in obj and obj["path"]:
                return Image.open(obj["path"])
            if "image" in obj and obj["image"] is not None:
                return Cityscapes._image_from_any(obj["image"])

        if isinstance(obj, str):
            return Image.open(obj)

        if isinstance(obj, (list, tuple)) and obj:
            return Cityscapes._image_from_any(obj[0])

        try:
            return Image.fromarray(obj)
        except Exception as e:
            raise TypeError(f"Unsupported image payload type: {type(obj)}") from e

    @staticmethod
    def _iter_nested_items(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                yield path, v
                yield from Cityscapes._iter_nested_items(v, path)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:3]):
                path = f"{prefix}[{i}]"
                yield path, v
                yield from Cityscapes._iter_nested_items(v, path)


# Backward compatibility for previous misspelling.
Citiscapes = Cityscapes
