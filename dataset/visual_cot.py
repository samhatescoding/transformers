from __future__ import annotations

import re
import ast
from io import BytesIO
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import urlopen

from datasets import load_dataset
from PIL import Image

from .hf_common import HFQADataset


class VisualCoT(HFQADataset):
    BOUNDING_BOX_REQUEST = re.compile(
        r"\s*Please provide the bounding box coordinate of the region that can "
        r"help you answer the question better\.?\s*$",
        re.IGNORECASE,
    )

    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "deepcs233/Visual-CoT") -> None:
        super().__init__(
            name="visual_cot",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            image_keys=("image", "image_path", "file_name"),
            question_keys=("question",),
            answer_keys=("answers", "answer", "full_answer"),
            data_files="viscot_363k.json",
        )

    def _standardize_row(self, row):
        out = dict(row)
        conversations = row.get("conversations")
        if isinstance(conversations, list):
            human_messages = [
                str(item.get("value", "")).replace("<image>", "").strip()
                for item in conversations
                if isinstance(item, dict) and item.get("from") == "human"
            ]
            model_messages = [
                str(item.get("value", "")).strip()
                for item in conversations
                if isinstance(item, dict) and item.get("from") == "gpt"
            ]
            out["question"] = self._without_bounding_box_request(
                human_messages[0] if human_messages else ""
            )
            out["answers"] = model_messages[-1:] if model_messages else []
        else:
            out["question"] = self._without_bounding_box_request(
                self.get_question_from_row(row)
            )
            out["answers"] = self.get_answers_from_row(row)
        return out

    def get_labels_img(self, row):
        return ["answer-relevant region"] if self.get_annotations_for_row(row) else []

    def get_annotations_for_row(self, row):
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            return []
        for item in conversations:
            if not isinstance(item, dict) or item.get("from") != "gpt":
                continue
            try:
                values = ast.literal_eval(str(item.get("value", "")).strip())
            except (SyntaxError, ValueError):
                continue
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(value) for value in values]
            except (TypeError, ValueError):
                continue
            if not all(0.0 <= value <= 1.0 for value in (x0, y0, x1, y1)):
                continue
            return [
                {
                    "label": "answer-relevant region",
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                }
            ]
        return []

    @classmethod
    def _without_bounding_box_request(cls, question: str) -> str:
        return cls.BOUNDING_BOX_REQUEST.sub("", str(question)).strip()

    def get_image_from_row(self, row):
        value = self._get_first_present(row, self.image_keys)
        if isinstance(value, list):
            if not value:
                raise ValueError("Visual-CoT row contains an empty image list.")
            value = value[0]
        if not isinstance(value, str):
            return self._coerce_image(value)
        url = (
            f"https://huggingface.co/datasets/{self.dataset_id}/resolve/main/"
            f"{quote(value.lstrip('/'), safe='/')}"
        )
        try:
            with urlopen(url, timeout=60) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        except HTTPError as exc:
            if exc.code != 404:
                raise
        return self._load_source_image(row, value)

    def _load_source_image(self, row, image_path: str) -> Image.Image:
        source_dataset = str(row.get("dataset", "")).strip().casefold()
        filename = image_path.split("###", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
        if source_dataset == "flickr30k":
            source_rows = load_dataset(
                "nlphuji/flickr30k",
                split="test",
                streaming=True,
                revision="refs/convert/parquet",
            )
            for source_row in source_rows:
                if str(source_row.get("filename", "")).strip() == filename:
                    return self._coerce_image(source_row["image"])
        raise ValueError(
            f"Visual-CoT image {image_path!r} is stored in the large archive and "
            f"no lightweight source adapter is available for {source_dataset!r}."
        )
