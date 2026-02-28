# data/flyingthings3d.py

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
from datasets import load_dataset

from .base import BaseDataset


class FlyingThings3D(BaseDataset):
    """
    FlyingThings3D adapter for image-based workflows.

    The dataset is sequence/stereo-oriented; this adapter selects one
    representative RGB image per row (prefers left-view image fields).

    Default HF repo:
      infinity1096/flyingthings3d_processed
    """

    DEFAULT_DATASET_ID = "infinity1096/flyingthings3d_processed"

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: Optional[str] = None,
        use_auth_token: Optional[bool] = None,
    ):
        self.name = "flyingthings3d"
        self.split = split
        self.streaming = streaming

        # Default to a known public repo if user didn't pass one.
        self.dataset_id = dataset_id or self.DEFAULT_DATASET_ID
        self.use_auth_token = use_auth_token

        self.ds = load_dataset(
            self.dataset_id,
            split=split,
            streaming=streaming,
            use_auth_token=use_auth_token,
        )
        print("Loaded dataset object. About to fetch first row...")
        first = next(iter(self.ds))
        print("Fetched first row keys:", list(first.keys()))

        self.class_labels: List[str] = self._extract_class_labels()
        self.labels = list(self.class_labels)

    def __repr__(self) -> str:
        return (
            f"<Dataset {self.name} | repo={self.dataset_id} | "
            f"split={self.split} | streaming={self.streaming}>"
        )

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.ds)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(self.ds):
            if i >= n:
                break
            samples.append(row)
        return samples

    # -------------------------
    # Labels
    # -------------------------
    def _extract_class_labels(self) -> List[str]:
        """
        Try to extract label names if the dataset exposes ClassLabel features.
        Many FlyingThings3D conversions won't have classification labels;
        this safely returns [] in that case.
        """
        try:
            feature_key = "label" if hasattr(self.ds, "features") and "label" in self.ds.features else "class"
            if hasattr(self.ds, "features") and feature_key in self.ds.features:
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
        return [label] if label else []

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

    # -------------------------
    # Images
    # -------------------------
    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        """
        Pick one representative image from a row.

        This tries common key names used by FlyingThings3D conversions.
        """
        image_key_order = (
            "left_image",
            "left",
            "image",
            "rgb",
            "frame",
            "img",
            "image_left",
            "left_img",
            "left_rgb",
        )

        for key in image_key_order:
            if key in row and row[key] is not None:
                return self._image_from_any(row[key]).convert("RGB")

        # Some datasets may store a list/sequence under "images"
        if "images" in row and row["images"]:
            first = row["images"][0]
            return self._image_from_any(first).convert("RGB")

        # Some datasets store stereo pairs under something like {"left": ..., "right": ...}
        # Attempt a lightweight fallback if present.
        if "stereo" in row and isinstance(row["stereo"], dict):
            for k in ("left", "left_image", "image_left"):
                if k in row["stereo"] and row["stereo"][k] is not None:
                    return self._image_from_any(row["stereo"][k]).convert("RGB")

        raise ValueError(
            "Row has no supported image field. Expected one of: "
            "left_image, left, image, rgb, frame, img, image_left, left_img, left_rgb, images, stereo."
        )

    @staticmethod
    def _image_from_any(obj: Any) -> Image.Image:
        """
        Convert common HF 'Image' payload shapes into a PIL Image.
        """
        if isinstance(obj, Image.Image):
            return obj

        # HF image dict-like: {"bytes": ..., "path": ...}
        if isinstance(obj, dict):
            if "bytes" in obj and obj["bytes"] is not None:
                return Image.open(BytesIO(obj["bytes"]))
            if "path" in obj and obj["path"]:
                return Image.open(obj["path"])

        # A direct filesystem path
        if isinstance(obj, str):
            return Image.open(obj)

        # A list/tuple: take first element
        if isinstance(obj, (list, tuple)) and obj:
            first = obj[0]
            if isinstance(first, Image.Image):
                return first
            try:
                return Image.fromarray(first)
            except Exception:
                pass

        # Numpy array-like fallback
        try:
            return Image.fromarray(obj)
        except Exception as e:
            raise TypeError(f"Unsupported image payload type: {type(obj)}") from e