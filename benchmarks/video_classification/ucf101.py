from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from PIL import Image

from dataset import UCF101

from ._video_classification import VideoClassificationBenchmark


class UCF101Benchmark(VideoClassificationBenchmark):
    dataset_cls = UCF101
    benchmark_name = "ucf101"
    default_split = "test"
    max_edge = 160

    def __init__(
        self,
        dataset=None,
        split: str = "test",
        streaming: bool = True,
        frames_per_clip: int = 4,
        search_limit: int = 512,
    ):
        super().__init__(dataset=dataset, split=split, streaming=streaming)
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
            representative = clip_rows[len(clip_rows) // 2]
            prepared_rows.append(
                {
                    "clip_id": key,
                    "video_id": clip_rows[0].get("video_id"),
                    "image": self._resize_image(self.dataset.get_image_from_row(representative)),
                    "label": clip_rows[0].get("label"),
                    "label_text": self.dataset.get_labels_img(clip_rows[0])[0],
                }
            )
            if len(prepared_rows) >= target:
                break
        rows = self._select_label_diverse_rows(prepared_rows, n)
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
        image = row.get("image")
        if image is None:
            raise ValueError("UCF101Benchmark row is missing its representative frame.")
        return self._coerce_image(image)

    def make_prompt(self, labels: List[str], row: Dict[str, Any] | None = None, image: Any | None = None) -> str:
        del row
        del image
        return (
            "USER: <image>\n"
            "The image is one representative frame from a video clip.\n"
            "Return exactly ONE action label from the complete list below.\n"
            "Complete label list:\n"
            f"{'\n'.join(labels)}\n"
            "ASSISTANT:"
        )
