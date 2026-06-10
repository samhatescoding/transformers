from __future__ import annotations

from typing import Any, Dict

from ..multiple_choice._multiple_choice import MultipleChoiceBenchmark


class ImagePreferenceBenchmark(MultipleChoiceBenchmark):
    """Choose the more aesthetically pleasing image from a pair."""

    task_type = "image_preference"
    fixed_choices = ("Image A", "Image B")
    default_instruction = "Choose the more aesthetically pleasing image."

    def get_image_for_row(self, row: Dict[str, Any]):
        left = row.get("image_a", row.get("source_image"))
        right = row.get("image_b", row.get("target_image"))
        if left is None or right is None:
            return super().get_image_for_row(row)
        return self._make_pair_canvas(
            self._coerce_image(left),
            self._coerce_image(right),
            "Image A",
            "Image B",
        )

    def make_prompt(self, labels, row=None, image: Any | None = None) -> str:
        del labels
        del image
        if row is None:
            raise ValueError("ImagePreferenceBenchmark requires a dataset row.")
        return (
            "USER: <image>\n"
            "Which image is more aesthetically pleasing overall?\n"
            "A. Image A\n"
            "B. Image B\n"
            "Return only A or B.\n"
            "ASSISTANT:"
        )
