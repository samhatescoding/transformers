from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List
from urllib.request import Request, urlopen

from PIL import Image

from .hf_common import HFMultipleChoiceSourceDataset


class OpenVid1M(HFMultipleChoiceSourceDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "nkp37/OpenVid-1M",
    ) -> None:
        super().__init__(
            name="openvid1m",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="image",
            image_keys=("image",),
            answer_keys=("prompt", "caption", "text"),
        )

    def _standardize_row(self, row):
        out = dict(row)
        video_name = str(row.get("video", ""))
        out["image"] = self._youtube_thumbnail_url(video_name[:11])
        out["question"] = self.get_question_from_row(row)
        out["answer"] = self.get_answer_from_row(row)
        out["caption"] = out["answer"]
        return out

    def get_captions_from_row(self, row):
        caption = str(row.get("caption", row.get("answer", ""))).strip()
        return [caption] if caption else []

    def get_available_samples(self, n: int) -> List[Dict[str, Any]]:
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
            raise ValueError("OpenVid thumbnail is unavailable.")
        row["image"] = loaded
        return loaded

    def _load_thumbnail(self, row: Dict[str, Any]) -> Image.Image | None:
        url = str(row.get("image", "")).strip()
        if not url:
            return None
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        except Exception:
            return None
