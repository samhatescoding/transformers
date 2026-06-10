from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from PIL import Image

from ._base_dataset import BaseDataset


class GQA(BaseDataset):
    def __init__(
        self,
        split: str = "validation",
        streaming: bool = True,
        dataset_id: str = "lmms-lab/GQA",
        image_config: str = "val_balanced_images",
        instruction_config: str = "val_balanced_instructions",
    ) -> None:
        self.name = "gqa"
        self.split = split
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.actual_split = "val"
        self.image_ds = load_dataset(dataset_id, image_config, split=self.actual_split, streaming=streaming)
        self.instruction_ds = load_dataset(dataset_id, instruction_config, split=self.actual_split, streaming=streaming)
        self.labels: List[str] = []

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.get_samples(64))

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        instructions: List[Dict[str, Any]] = []
        needed_image_ids = set()
        for row in self.instruction_ds:
            image_id = str(row.get("imageId", "")).strip()
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if (
                not image_id
                or not question
                or not answer
                or image_id in needed_image_ids
            ):
                continue
            instructions.append(dict(row))
            needed_image_ids.add(image_id)
            if len(instructions) >= n:
                break

        images_by_id: Dict[str, Image.Image] = {}
        for row in self.image_ds:
            image_id = str(row.get("id", "")).strip()
            if image_id in needed_image_ids:
                images_by_id[image_id] = row["image"].convert("RGB")
                if len(images_by_id) >= len(needed_image_ids):
                    break

        samples: List[Dict[str, Any]] = []
        for row in instructions:
            image_id = str(row.get("imageId", "")).strip()
            image = images_by_id.get(image_id)
            if image is None:
                continue
            samples.append(
                {
                    "id": row.get("id"),
                    "image_id": image_id,
                    "image": image,
                    "question": str(row.get("question", "")).strip(),
                    "answers": [str(row.get("answer", "")).strip()],
                }
            )
            if len(samples) >= n:
                break
        return samples

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_labels_img(self, row) -> List[str]:
        return list(row.get("answers", []))

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["image"].convert("RGB")

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        return str(row.get("question", "")).strip()

    def get_answers_from_row(self, row: Dict[str, Any]) -> List[str]:
        return [str(answer).strip() for answer in row.get("answers", []) if str(answer).strip()]
