# data/flickr30k.py

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Set
from PIL import Image
from datasets import load_dataset

from .base import BaseDataset


class Flickr30k(BaseDataset):
    """
    Flickr30k on Hugging Face (caption dataset).
    - Supports streaming
    - Provides caption access
    - Provides noun/label extraction from captions
    """

    def __init__(self, split: str = "test", streaming: bool = True):
        self.name = "flickr30k"
        self.split = split
        self.streaming = streaming

        self.ds = load_dataset(
            "nlphuji/flickr30k",
            split=split,
            streaming=streaming,
            revision="refs/convert/parquet",
        )

        # Will be filled later by your sampling pipeline
        self.labels: List[str] = []

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
        if "image" not in row:
            raise ValueError("Row has no 'image' field.")

        img = row["image"]

        if isinstance(img, Image.Image):
            return img.convert("RGB")

        # Sometimes HF image is dict-like
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                from io import BytesIO
                return Image.open(BytesIO(img["bytes"])).convert("RGB")
            if "path" in img and img["path"]:
                return Image.open(img["path"]).convert("RGB")

        raise TypeError(f"Unsupported image type in row['image']: {type(img)}")

    def get_captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        # Common variants
        if "caption" in row and row["caption"] is not None:
            if isinstance(row["caption"], list):
                return [str(x) for x in row["caption"]]
            return [str(row["caption"])]

        if "sentences" in row and row["sentences"] is not None:
            if isinstance(row["sentences"], list):
                return [str(x) for x in row["sentences"]]
            return [str(row["sentences"])]

        # Fallback: try anything that looks like captions
        for key in ("captions", "text", "description"):
            if key in row and row[key] is not None:
                if isinstance(row[key], list):
                    return [str(x) for x in row[key]]
                return [str(row[key])]

        return []

    def get_labels_img(self, row) -> List[str]:
        captions = self.get_captions_from_row(row)
        nouns_this_image = self.extract_nouns(captions)
        return sorted(nouns_this_image)

    def get_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        """
        Given selected rows, build a sorted unique label list of nouns
        from all captions across those rows.
        """
        all_captions: List[str] = []
        for row in rows:
            all_captions.extend(self.get_captions_from_row(row))

        nouns = self.extract_nouns(all_captions)

        # Sort for determinism (nice for debugging)
        labels = sorted(nouns)
        self.labels = labels
        return labels
