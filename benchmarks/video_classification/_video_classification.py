from __future__ import annotations

from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw

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

    def _make_contact_sheet(self, frames: Sequence[Image.Image]) -> Image.Image:
        rgb_frames = [frame.convert("RGB") for frame in frames]
        tile_width = max(frame.width for frame in rgb_frames)
        tile_height = max(frame.height for frame in rgb_frames)
        canvas = Image.new("RGB", (tile_width * len(rgb_frames), tile_height + 24), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for idx, frame in enumerate(rgb_frames, start=1):
            x = (idx - 1) * tile_width
            canvas.paste(frame, (x, 24))
            draw.text((x + 8, 4), f"Frame {idx}", fill=(0, 0, 0))
        return canvas

    def _coerce_image(self, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if hasattr(value, "convert"):
            try:
                return value.convert("RGB")
            except Exception:
                pass
        return Image.fromarray(value)
