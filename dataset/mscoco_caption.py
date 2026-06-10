from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from datasets import load_dataset
from PIL import Image

from ._base_dataset import BaseDataset


class MSCOCOCaption(BaseDataset):
    IMAGE_DOWNLOAD_ATTEMPTS = 5
    RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        split: str = "test",
        streaming: bool = True,
        dataset_id: str = "yerevann/coco-karpathy",
    ) -> None:
        self.name = "mscoco_caption"
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
        del rows
        return []

    def get_labels_img(self, row) -> List[str]:
        return []

    def get_captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        captions = row.get("captions")
        if isinstance(captions, list):
            return [str(item).strip() for item in captions if str(item).strip()]
        sentences = row.get("sentences")
        if isinstance(sentences, list):
            return [str(item).strip() for item in sentences if str(item).strip()]
        return []

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        url = str(row.get("url", "")).strip()
        if not url:
            raise ValueError("MSCOCO caption row is missing an image URL.")
        parts = urlsplit(url)
        if parts.scheme == "http" and parts.netloc == "images.cocodataset.org":
            url = urlunsplit(
                ("https", parts.netloc, parts.path, parts.query, parts.fragment)
            )
        request = Request(url, headers={"User-Agent": "transformers-benchmark/1.0"})
        for attempt in range(self.IMAGE_DOWNLOAD_ATTEMPTS):
            try:
                with urlopen(request, timeout=30) as response:
                    return Image.open(BytesIO(response.read())).convert("RGB")
            except HTTPError as exc:
                if (
                    exc.code not in self.RETRYABLE_HTTP_CODES
                    or attempt == self.IMAGE_DOWNLOAD_ATTEMPTS - 1
                ):
                    raise
            except (URLError, TimeoutError):
                if attempt == self.IMAGE_DOWNLOAD_ATTEMPTS - 1:
                    raise
            time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["captions"] = self.get_captions_from_row(row)
        return out
