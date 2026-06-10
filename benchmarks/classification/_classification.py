from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from .._base_benchmark import BaseBenchmark
from ..type_l_labels import get_complete_type_l_labels


class ClassificationBenchmark(BaseBenchmark):
    max_edge = 336

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

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del row
        del image
        return (
            "USER: <image>\n"
            "Return exactly ONE label from the complete list below "
            "(one item only, no extra words).\n"
            "Complete label list:\n"
            f"{'\n'.join(labels)}\n"
            "ASSISTANT:"
        )

    def get_image_for_row(self, row):
        image = self.dataset.get_image_from_row(row)
        if not isinstance(image, Image.Image):
            return image
        return self._resize_image(image)
