from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List
from urllib.request import Request, urlopen

from PIL import Image

from .hf_common import HFClassificationDataset


class Kinetics(HFClassificationDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "iejMac/CLIP-Kinetics700",
    ) -> None:
        super().__init__(
            name="kinetics",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            label_keys=("label",),
        )

    def _standardize_row(self, row):
        out = dict(row)
        out["image"] = self._youtube_thumbnail_url(row.get("youtube_id"))
        return out

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        seen_images = set()
        for raw_row in self.ds:
            row = self._standardize_row(raw_row)
            image_url = str(row.get("image", "")).strip()
            if not image_url or image_url in seen_images:
                continue
            seen_images.add(image_url)
            image = self._load_thumbnail(row)
            if image is None:
                continue
            row["image"] = image
            samples.append(row)
            if len(samples) >= n:
                break
        return samples

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        loaded = self._load_thumbnail(row)
        if loaded is None:
            raise ValueError("Kinetics thumbnail is unavailable.")
        row["image"] = loaded
        return loaded

    def _load_thumbnail(self, row: Dict[str, Any]) -> Image.Image | None:
        url = str(row.get("image", "")).strip()
        if not url:
            return None
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=10) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        except Exception:
            return None
