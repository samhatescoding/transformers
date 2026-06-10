from __future__ import annotations

from typing import Any, Dict, Iterable, List

from PIL import Image
from datasets import load_dataset

from ._base_dataset import BaseDataset


class BLIP3o60k(BaseDataset):
    """Stream BLIP3o-60k directly from its official WebDataset archives."""

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "BLIP3o/BLIP3o-60k",
        data_files: str | List[str] | None = None,
    ) -> None:
        self.name = "blip3o_60k"
        self.split = split
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.labels: List[str] = []
        archive_files = data_files or f"hf://datasets/{dataset_id}/*.tar"
        self.ds = load_dataset("webdataset", data_files=archive_files, split=split, streaming=streaming)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for row in self.ds:
            yield self._standardize_row(row)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(self.ds):
            if index >= n:
                break
            rows.append(self._standardize_row(row))
        return rows

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        answer = self.get_answer_from_row(row)
        return [answer] if answer else []

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        value = row.get("image")
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        raise ValueError("BLIP3o-60k row does not contain a decoded image.")

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        del row
        return "Which prompt was used to generate this image?"

    def get_answer_from_row(self, row: Dict[str, Any]) -> str:
        return str(row.get("answer", row.get("text", row.get("txt", "")))).strip()

    def _standardize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        image = next(
            (row[key] for key in ("jpg", "jpeg", "png", "webp", "image") if row.get(key) is not None),
            None,
        )
        if image is None:
            raise ValueError("BLIP3o-60k WebDataset row is missing an image field.")
        out["image"] = image.convert("RGB") if isinstance(image, Image.Image) else image
        out["answer"] = str(
            row.get("text", row.get("txt", row.get("prompt", row.get("instruction", ""))))
        ).strip()
        out["question"] = self.get_question_from_row(out)
        return out
