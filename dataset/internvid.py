from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List
from urllib.request import Request, urlopen

from PIL import Image

from .hf_common import HFVideoCaptionDataset


class InternVid(HFVideoCaptionDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "OpenGVLab/InternVid-Full",
    ) -> None:
        super().__init__(
            name="internvid",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            frame_keys=("image",),
            max_frames=1,
            caption_keys=("captions", "Caption"),
        )
        self._sample_iterator = None
        self._preview_progress = None

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        seen_video_ids = set()
        current_video_id = ""
        current_rows: List[Dict[str, Any]] = []
        for row in self.ds:
            video_id = self._video_id(row)
            if not video_id or video_id in seen_video_ids:
                continue
            if current_video_id and video_id != current_video_id:
                seen_video_ids.add(current_video_id)
                yield self._merge_video_rows(current_video_id, current_rows)
                current_rows = []
            current_video_id = video_id
            current_rows.append(row)

        if current_video_id and current_rows:
            yield self._merge_video_rows(current_video_id, current_rows)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        self._sample_iterator = iter(self)
        for index in range(n):
            try:
                samples.append(next(self._sample_iterator))
            except StopIteration:
                break
            preview_progress = getattr(self, "_preview_progress", None)
            if callable(preview_progress):
                preview_progress(
                    f"Grouped captions for {index + 1} of {n} distinct InternVid videos."
                )
        return samples

    def set_preview_progress_callback(self, callback) -> None:
        self._preview_progress = callback

    def get_next_available_sample(self) -> Dict[str, Any]:
        if self._sample_iterator is None:
            self._sample_iterator = iter(self)
        for row in self._sample_iterator:
            image = self._load_thumbnail(row)
            if image is None:
                continue
            row["image"] = image
            return row
        raise ValueError("No additional InternVid row with an available thumbnail was found.")

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        image = self._load_thumbnail(row)
        if image is None:
            raise ValueError(f"InternVid thumbnail is unavailable for {row.get('video_id', 'unknown video')}.")
        row["image"] = image
        return image

    def _load_thumbnail(self, row: Dict[str, Any]) -> Image.Image | None:
        url = str(row.get("image", "")).strip()
        if not url:
            return None
        request = Request(
            url,
            headers={"User-Agent": "transformers-benchmark-input-browser/1.0"},
        )
        try:
            with urlopen(request, timeout=4) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        except Exception:
            return None

    def _standardize_row(self, row):
        out = super()._standardize_row(row)
        video_id = self._video_id(row)
        out["video_id"] = video_id
        out["image"] = self._youtube_thumbnail_url(video_id)
        return out

    def _merge_video_rows(
        self,
        video_id: str,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out = self._standardize_row(rows[0])
        captions: List[str] = []
        seen = set()
        for row in rows:
            for caption in self.get_captions_from_row(row):
                normalized = self.normalize_text(caption)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                captions.append(caption)
        out["video_id"] = video_id
        out["captions"] = captions
        return out

    @staticmethod
    def _video_id(row: Dict[str, Any]) -> str:
        return str(row.get("YoutubeID", "")).strip()
