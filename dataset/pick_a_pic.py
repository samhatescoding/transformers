from __future__ import annotations

from typing import Any, Dict, Sequence

from PIL import Image

from .hf_common import HFBaseDataset


class PickAPic(HFBaseDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "kevinkingslin/pick-a-pic",
        image_a_keys: Sequence[str] = ("image0", "jpg_0", "image_0"),
        image_b_keys: Sequence[str] = ("image1", "jpg_1", "image_1"),
        preference_keys: Sequence[str] = ("human_pref", "preference", "label"),
    ) -> None:
        super().__init__(
            name="pick_a_pic",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
        )
        self.image_a_keys = tuple(image_a_keys)
        self.image_b_keys = tuple(image_b_keys)
        self.preference_keys = tuple(preference_keys)

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["image_a"] = self._extract_image(row, self.image_a_keys)
        out["image_b"] = self._extract_image(row, self.image_b_keys)
        out["question"] = "Which image is more aesthetically pleasing?"
        out["answer"] = self.get_answer_from_row(row)
        out["choices"] = ["Image A", "Image B"]
        return out

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        value = row.get("image_a")
        if value is not None:
            return self._coerce_image(value)
        return self._extract_image(row, self.image_a_keys)

    def get_labels(self, rows) -> list[str]:
        del rows
        return []

    def get_labels_img(self, row: Dict[str, Any]) -> list[str]:
        answer = self.get_answer_from_row(row)
        return [answer] if answer else []

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        del row
        return "Which image is more aesthetically pleasing?"

    def get_choices_from_row(self, row: Dict[str, Any]) -> list[str]:
        del row
        return ["Image A", "Image B"]

    def get_answer_from_row(self, row: Dict[str, Any]) -> str:
        existing = row.get("answer")
        if existing in ("Image A", "Image B"):
            return str(existing)

        preference = self._get_first_present(row, self.preference_keys)
        if preference is not None:
            try:
                index = int(preference)
            except (TypeError, ValueError):
                index = -1
            if index in (0, 1):
                return "Image A" if index == 0 else "Image B"

        label_a = row.get("label_0")
        label_b = row.get("label_1")
        if label_a is not None and label_b is not None:
            score_a = float(label_a)
            score_b = float(label_b)
            if score_a != score_b:
                return "Image A" if score_a > score_b else "Image B"
        return ""
