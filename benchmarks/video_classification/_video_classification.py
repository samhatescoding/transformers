from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from .._base_benchmark import BaseBenchmark
from ..type_l_labels import get_complete_type_l_labels


class VideoClassificationBenchmark(BaseBenchmark):
    def prepare(self, n: int, label_sample_size: int):
        return self._prepare_label_diverse_rows(n=n, label_sample_size=label_sample_size)

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        complete_labels = get_complete_type_l_labels(self)
        if complete_labels is not None:
            return complete_labels
        return super().get_candidate_labels(rows)

    def get_prompt_labels_for_row(
        self,
        row: Dict[str, Any],
        labels: List[str],
    ) -> List[str]:
        complete_labels = get_complete_type_l_labels(self)
        if complete_labels is not None:
            return complete_labels
        return super().get_prompt_labels_for_row(row, labels)

    def get_image_for_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("frames"):
            frames = row["frames"]
            return self._coerce_image(frames[len(frames) // 2])
        return self.dataset.get_image_from_row(row)

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        getter = getattr(self.dataset, "get_labels_img", None)
        if callable(getter):
            labels = list(getter(row))
            if labels:
                return labels
        label_text = str(row.get("label_text", "")).strip()
        return [label_text] if label_text else []

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del row
        del image
        label_text = "\n".join(labels)
        return (
            "USER: <image>\n"
            "The image is one representative frame from a video clip.\n"
            "Return exactly ONE action label from the complete list below.\n"
            "Complete label list:\n"
            f"{label_text}\n"
            "ASSISTANT:"
        )
