# data/ucf101.py

from __future__ import annotations

from typing import Any, Dict, Iterable, List
from PIL import Image
from datasets import load_dataset

from .base import BaseDataset


class UCF101(BaseDataset):
    """
    UCF101 (frames-as-images) from Hugging Face: flwrlabs/ucf101

    Dataset rows look like:
      {
        "image": PIL.Image,
        "video_id": str,
        "clip_id": str,
        "frame": int,
        "label": int
      }

    Splits: "train", "test"
    """

    def __init__(self, split: str = "train", streaming: bool = True):
        self.name = "ucf101"
        self.split = split
        self.streaming = streaming

        # IMPORTANT: dataset id is namespaced on the hub
        self.ds = load_dataset(
            "flwrlabs/ucf101",
            split=split,
            streaming=streaming,
        )

        # Class names
        self.labels: List[str] = []
        try:
            self.labels = list(self.ds.features["label"].names)  # type: ignore[attr-defined]
        except Exception:
            # In some streaming contexts features may not be available immediately
            self.labels = []

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

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        img = row.get("image")
        if img is None:
            raise ValueError("Row has no 'image' field.")
        if not isinstance(img, Image.Image):
            raise TypeError(f"row['image'] is not a PIL Image: {type(img)}")
        return img.convert("RGB")

    def get_labels_img(self, row) -> List[str]:
        # One label per frame (action class of the originating clip)
        idx = int(row["label"])
        if self.labels and 0 <= idx < len(self.labels):
            return [self.labels[idx]]
        # Fallback if labels list not available:
        return [str(idx)]

    def get_labels(self, rows) -> List[str]:
        return self.labels