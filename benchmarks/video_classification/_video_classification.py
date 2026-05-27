from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from .._base_benchmark import BaseBenchmark


class VideoClassificationBenchmark(BaseBenchmark):
    def get_image_for_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("frames"):
            return self._make_contact_sheet([self._coerce_image(frame) for frame in row["frames"]])
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
        label_text = ", ".join(labels)
        return (
            "USER: <image>\n"
            "The image shows multiple frames from the same video clip.\n"
            "Return exactly ONE label from this list:\n"
            f"{label_text}\n"
            "ASSISTANT:"
        )
