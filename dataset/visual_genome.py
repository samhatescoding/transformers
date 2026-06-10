from __future__ import annotations

from PIL import ImageDraw

from .hf_common import HFQADataset


class VisualGenome(HFQADataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "dipta007/bengali-visual-genome-1.0-prompt",
    ) -> None:
        super().__init__(
            name="visual_genome",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("img_context",),
            question_keys=("question",),
            answer_keys=("Eng",),
        )

    def get_question_from_row(self, row):
        return "What is shown in the highlighted region of the image?"

    def get_image_from_row(self, row):
        image = self._extract_image(row, self.image_keys)
        try:
            x = int(row["X"])
            y = int(row["Y"])
            width = int(row["W"])
            height = int(row["H"])
        except (KeyError, TypeError, ValueError):
            return image

        highlighted = image.copy()
        draw = ImageDraw.Draw(highlighted)
        line_width = max(3, round(min(highlighted.size) / 150))
        draw.rectangle(
            (x, y, x + width, y + height),
            outline=(255, 0, 0),
            width=line_width,
        )
        return highlighted
