from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from dataset import UCF101

from ._video_classification import VideoClassificationBenchmark


class UCF101Benchmark(VideoClassificationBenchmark):
    benchmark_name = "ucf101"
    default_max_new_tokens = 16
    max_edge = 160

    def __init__(
        self,
        dataset=None,
        split: str = "test",
        streaming: bool = True,
        frames_per_clip: int = 4,
        search_limit: int = 512,
    ):
        dataset = dataset or UCF101(split=split, streaming=streaming)
        super().__init__(dataset=dataset, name=self.benchmark_name)
        self.frames_per_clip = max(2, int(frames_per_clip))
        self.search_limit = max(self.frames_per_clip, int(search_limit))

    def prepare(self, n: int, label_sample_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        target = max(n, label_sample_size)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        ordered_groups: List[str] = []
        for idx, row in enumerate(self.dataset):
            if idx >= self.search_limit and len(ordered_groups) >= target:
                break
            key = str(row.get("clip_id") or row.get("video_id") or f"row-{idx}")
            if key not in grouped:
                ordered_groups.append(key)
            grouped[key].append(row)
        prepared_rows: List[Dict[str, Any]] = []
        for key in ordered_groups:
            clip_rows = grouped[key]
            if len(clip_rows) < self.frames_per_clip:
                continue
            selected = self._sample_clip_frames(clip_rows)
            prepared_rows.append(
                {
                    "clip_id": key,
                    "video_id": clip_rows[0].get("video_id"),
                    "frames": [self._resize_image(self.dataset.get_image_from_row(frame_row)) for frame_row in selected],
                    "label": clip_rows[0].get("label"),
                    "label_text": self.dataset.get_labels_img(clip_rows[0])[0],
                }
            )
            if len(prepared_rows) >= target:
                break
        rows = prepared_rows[:n]
        labels = self.get_candidate_labels(prepared_rows[:target])
        return rows, labels

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        labels: List[str] = []
        seen = set()
        for row in rows:
            label = str(row.get("label_text", "")).strip()
            if not label:
                continue
            normalized = self.dataset.normalize_text(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            labels.append(label)
        return labels or list(getattr(self.dataset, "labels", []))

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        label = str(row.get("label_text", "")).strip()
        return [label] if label else []

    def get_image_for_row(self, row: Dict[str, Any]) -> Image.Image:
        frames = row.get("frames") or []
        if not frames:
            raise ValueError("UCF101Benchmark row is missing frames.")
        return self._make_contact_sheet(frames)

    def make_prompt(self, labels: List[str], row: Dict[str, Any] | None = None, image: Any | None = None) -> str:
        del row
        del image
        return (
            "USER: <image>\n"
            "The image shows multiple frames from the same video clip.\n"
            "Return exactly ONE action label from this list:\n"
            f"{', '.join(labels)}\n"
            "ASSISTANT:"
        )

    def _sample_clip_frames(self, clip_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(clip_rows) <= self.frames_per_clip:
            return clip_rows[: self.frames_per_clip]
        step = (len(clip_rows) - 1) / (self.frames_per_clip - 1)
        indices = [round(step * i) for i in range(self.frames_per_clip)]
        return [clip_rows[i] for i in indices]

    def _make_contact_sheet(self, frames: List[Image.Image]) -> Image.Image:
        normalized = [frame.convert("RGB") for frame in frames]
        tile_width = max(frame.width for frame in normalized)
        tile_height = max(frame.height for frame in normalized)
        canvas = Image.new("RGB", (tile_width * len(normalized), tile_height + 24), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for idx, frame in enumerate(normalized, start=1):
            x = (idx - 1) * tile_width
            canvas.paste(frame, (x, 24))
            draw.text((x + 8, 4), f"Frame {idx}", fill=(0, 0, 0))
        return canvas

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        largest_edge = max(width, height)
        if largest_edge <= self.max_edge:
            return image
        scale = self.max_edge / float(largest_edge)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, Image.Resampling.BICUBIC)
