from __future__ import annotations

import json
import tarfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

from huggingface_hub import HfFileSystem
from PIL import Image

from ._base_dataset import BaseDataset


class FlyingThings3D(BaseDataset):
    archive_name = "flyingthings3d__frames_cleanpass.tar_part0"

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: str = "ssbai/flyingthings3d",
    ) -> None:
        self.name = "flyingthings3d"
        self.dataset_id = dataset_id
        self.split = split
        self.streaming = streaming
        self.labels: List[str] = []
        self._rows: List[Dict[str, Any]] = []

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.get_samples(64))

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        if n <= len(self._rows):
            return self._rows[:n]

        local_rows = self._get_local_samples(n)
        if local_rows:
            self._rows = local_rows
            return self._rows[:n]

        pairs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        fs = HfFileSystem(token=True)
        archive_path = f"datasets/{self.dataset_id}/{self.archive_name}"
        with fs.open(
            archive_path,
            "rb",
            block_size=4 * 1024 * 1024,
            cache_type="readahead",
        ) as remote_file:
            with tarfile.open(fileobj=remote_file, mode="r:") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".png"):
                        continue
                    parts = member.name.replace("\\", "/").split("/")
                    if len(parts) < 3 or parts[-2] not in {"left", "right"}:
                        continue

                    view = parts[-2]
                    pair_key = "/".join(parts[:-2] + [parts[-1]])
                    pair = pairs.setdefault(pair_key, {})
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    pair[view] = Image.open(BytesIO(extracted.read())).convert("RGB")
                    if "left" not in pair or "right" not in pair:
                        continue

                    self._rows.append(
                        {
                            "id": pair_key,
                            "source_image": pair["left"],
                            "target_image": pair["right"],
                            "question": "Which panel shows the right-camera view of the stereo scene?",
                            "answer": "right panel",
                            "choices": ["left panel", "right panel"],
                        }
                    )
                    del pairs[pair_key]
                    if len(self._rows) >= n:
                        return self._rows[:n]

        return self._rows[:n]

    def get_samples_at_indices(self, indices: Iterable[int]) -> List[Dict[str, Any]]:
        requested = sorted(set(int(index) for index in indices))
        if not requested or requested[0] < 0:
            return []

        requested_set = set(requested)
        selected: Dict[int, Dict[str, Any]] = {}
        pairs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        pair_index = 0
        fs = HfFileSystem(token=True)
        archive_path = f"datasets/{self.dataset_id}/{self.archive_name}"
        with fs.open(
            archive_path,
            "rb",
            block_size=4 * 1024 * 1024,
            cache_type="readahead",
        ) as remote_file:
            with tarfile.open(fileobj=remote_file, mode="r:") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".png"):
                        continue
                    parts = member.name.replace("\\", "/").split("/")
                    if len(parts) < 3 or parts[-2] not in {"left", "right"}:
                        continue

                    view = parts[-2]
                    pair_key = "/".join(parts[:-2] + [parts[-1]])
                    pair = pairs.setdefault(pair_key, {})
                    if pair_index in requested_set:
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            continue
                        pair[view] = Image.open(BytesIO(extracted.read())).convert("RGB")
                    else:
                        pair[view] = True

                    if "left" not in pair or "right" not in pair:
                        continue
                    if pair_index in requested_set:
                        selected[pair_index] = self._make_row(
                            pair_key,
                            pair["left"],
                            pair["right"],
                        )
                    del pairs[pair_key]
                    if pair_index >= requested[-1]:
                        break
                    pair_index += 1

        return [selected[index] for index in requested if index in selected]

    @staticmethod
    def _make_row(pair_key: str, left: Image.Image, right: Image.Image) -> Dict[str, Any]:
        return {
            "id": pair_key,
            "source_image": left,
            "target_image": right,
            "question": "Which panel shows the right-camera view of the stereo scene?",
            "answer": "right panel",
            "choices": ["left panel", "right panel"],
        }

    def _get_local_samples(self, n: int) -> List[Dict[str, Any]]:
        repository_root = Path(__file__).resolve().parents[1]
        choices_path = (
            repository_root / "benchmark_choices" / "type_a" / "flyingthings3d.json"
        )
        if not choices_path.exists():
            return []

        payload = json.loads(choices_path.read_text(encoding="utf-8"))
        choices_root = choices_path.parent
        rows = []
        for annotation in payload.get("rows", [])[:n]:
            left_path = choices_root / annotation["left_image"]
            right_path = choices_root / annotation["right_image"]
            if not left_path.exists() or not right_path.exists():
                return []
            rows.append(
                {
                    "id": annotation["source_id"],
                    "source_image": Image.open(left_path).convert("RGB"),
                    "target_image": Image.open(right_path).convert("RGB"),
                    "question": "Which panel shows the right-camera view of the stereo scene?",
                    "answer": "right panel",
                    "choices": ["left panel", "right panel"],
                }
            )
        return rows

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        return row["source_image"].convert("RGB")

    def get_labels(self, rows) -> List[str]:
        del rows
        return []

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        return [str(row["answer"])]

    def get_question_from_row(self, row: Dict[str, Any]) -> str:
        return str(row["question"])

    def get_answer_from_row(self, row: Dict[str, Any]) -> str:
        return str(row["answer"])

    def get_choices_from_row(self, row: Dict[str, Any]) -> List[str]:
        return [str(choice) for choice in row["choices"]]
