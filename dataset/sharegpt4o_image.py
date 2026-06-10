from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image
from datasets import load_dataset

from ._base_dataset import BaseDataset
from .hf_common import HFMultipleChoiceSourceDataset


class ShareGPT4oImage(BaseDataset):
    """ShareGPT-4o-Image from a streamable WebDataset mirror or local release."""

    DEFAULT_DATASET_ID = "hanlincs/sharegpt4oimage_processed"

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        data_dir: str | os.PathLike[str] | None = None,
        subset: str = "text_to_image",
        dataset_id: str = DEFAULT_DATASET_ID,
    ) -> None:
        if subset not in {"text_to_image", "text_and_image_to_image"}:
            raise ValueError("subset must be 'text_to_image' or 'text_and_image_to_image'.")
        self.name = "sharegpt4o_image"
        self.split = split
        self.subset = subset
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.labels: List[str] = []
        root = data_dir or os.environ.get("SHAREGPT4O_IMAGE_ROOT")
        self.data_dir = Path(root).expanduser() if root else None
        self.rows: List[Dict[str, Any]] | None = None

        if self.data_dir is not None:
            if not self.data_dir.exists():
                raise FileNotFoundError(f"ShareGPT-4o-Image data directory not found: {self.data_dir}")
            metadata_path = self.data_dir / f"{subset}.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"ShareGPT-4o-Image metadata file not found: {metadata_path}")
            self.rows = self._load_local_rows(metadata_path)
            self.ds = None
        else:
            if subset != "text_to_image":
                raise ValueError("The streamable mirror currently supports only the text_to_image subset.")
            self.ds = load_dataset(
                dataset_id,
                split=split,
                streaming=streaming,
                token=os.getenv("HF_TOKEN"),
            )

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if self.rows is not None:
            return iter(self.rows)
        assert self.ds is not None
        return (self._standardize_hf_row(row) for row in self.ds)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        if self.rows is not None:
            return self.rows[:n]
        samples: List[Dict[str, Any]] = []
        for index, row in enumerate(self):
            if index >= n:
                break
            samples.append(row)
        return samples

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
        if row.get("image_path"):
            return Image.open(row["image_path"]).convert("RGB")
        raise ValueError("ShareGPT-4o-Image row does not contain an image.")

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        del row
        return "Which prompt was used to generate this image?"

    def get_answer_from_row(self, row: Dict[str, Any]) -> str:
        return str(row.get("answer", row.get("txt", ""))).strip()

    def _standardize_hf_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["image"] = row.get("jpg", row.get("image"))
        out["answer"] = str(row.get("txt", row.get("prompt", ""))).strip()
        out["question"] = self.get_question_from_row(out)
        return out

    def _load_local_rows(self, metadata_path: Path) -> List[Dict[str, Any]]:
        assert self.data_dir is not None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("data", payload.get("samples", []))
        rows: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            prompt = self._first_text(record, ("input_prompt", "prompt", "instruction", "caption", "text"))
            image_value = self._first_text(
                record,
                ("image", "image_path", "output_image", "target_image", "generated_image"),
            )
            image_path = self._resolve_path(image_value)
            if not prompt or image_path is None:
                continue
            row = dict(record)
            row["image_path"] = str(image_path)
            row["answer"] = prompt
            row["question"] = self.get_question_from_row(row)
            rows.append(row)
        if not rows:
            raise ValueError(f"No usable ShareGPT-4o-Image records were found in {metadata_path}.")
        return rows

    def _resolve_path(self, value: str) -> Path | None:
        if not value or self.data_dir is None:
            return None
        candidate = Path(value)
        paths = [candidate] if candidate.is_absolute() else [self.data_dir / candidate]
        for path in paths:
            if path.exists():
                return path
        matches = list(self.data_dir.glob(f"**/{candidate.name}"))
        return matches[0] if matches else None

    @staticmethod
    def _first_text(row: Dict[str, Any], keys) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""


class ShareGPT4oImageEdit(HFMultipleChoiceSourceDataset):
    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "svjack/ShareGPT-4o-Image-Text-Edit",
    ) -> None:
        super().__init__(
            name="sharegpt4o_image_edit",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            mode="pair",
            source_image_keys=("input_image",),
            target_image_keys=("output_image",),
            question_keys=("input_prompt",),
            answer_keys=("input_prompt",),
        )
