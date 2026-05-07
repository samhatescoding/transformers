from __future__ import annotations

from io import BytesIO
from urllib.request import urlopen

from PIL import Image

from .hf_common import HFMultipleChoiceSourceDataset


class ConceptualCaptions(HFMultipleChoiceSourceDataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "google-research-datasets/conceptual_captions") -> None:
        super().__init__(
            name="conceptual_captions",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image",),
            answer_keys=("caption", "captions", "text"),
        )

    def _standardize_row(self, row):
        out = dict(row)
        out["image_url"] = str(row.get("image_url", "")).strip()
        out["question"] = self.get_question_from_row(row)
        out["answer"] = self.get_answer_from_row(row)
        out["captions"] = self.get_captions_from_row(row)
        return out

    def get_captions_from_row(self, row):
        answer = self.get_answer_from_row(row)
        return [answer] if answer else []

    def get_image_from_row(self, row) -> Image.Image:
        if row.get("image") is not None:
            return super().get_image_from_row(row)
        image_url = str(row.get("image_url", "")).strip()
        if not image_url:
            raise ValueError("Conceptual Captions row is missing image content.")
        with urlopen(image_url, timeout=30) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
