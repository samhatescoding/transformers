from __future__ import annotations

import csv
import io
import os
import random
import zipfile
from bisect import bisect_left
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image

from ._base_dataset import BaseDataset


class TAD66K(BaseDataset):
    """TAD66K from local files or range-streamed images with official scores."""

    IMAGE_KEYS = ("image", "image_path", "path", "filename", "file_name", "name")
    SCORE_KEYS = ("score", "aesthetic_score", "mos", "MOS", "mean_score", "label")
    DEFAULT_DATASET_ID = "Shuai1995/TAD66K_for_Image_Aesthetics_Assessment"
    DEFAULT_IMAGE_ARCHIVE_URL = (
        "https://huggingface.co/datasets/"
        "Shuai1995/TAD66K_for_Image_Aesthetics_Assessment/resolve/main/TAD66K.zip"
    )

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        data_dir: str | os.PathLike[str] | None = None,
        metadata_file: str | os.PathLike[str] | None = None,
        image_keys: Sequence[str] = IMAGE_KEYS,
        score_keys: Sequence[str] = SCORE_KEYS,
        dataset_id: str = DEFAULT_DATASET_ID,
        image_archive_url: str = DEFAULT_IMAGE_ARCHIVE_URL,
    ) -> None:
        self.name = "tad66k"
        self.split = split
        self.streaming = streaming
        self.image_keys = tuple(image_keys)
        self.score_keys = tuple(score_keys)
        self.labels = [str(value) for value in range(1, 11)]
        self.dataset_id = dataset_id
        self.image_archive_url = image_archive_url
        self._remote_zip = None
        root = data_dir or os.environ.get("TAD66K_ROOT")
        self.data_dir = Path(root).expanduser() if root else None

        if self.data_dir is not None:
            if not self.data_dir.exists():
                raise FileNotFoundError(f"TAD66K data directory not found: {self.data_dir}")
            self.metadata_path = self._find_metadata_file(metadata_file)
            self.rows = self._load_local_rows()
        else:
            self.metadata_path = None
            self.rows = self._load_remote_scores()

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.rows)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        return self.rows[:n]

    def get_score_spaced_samples(self, n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            return []

        scored_rows = sorted(
            (
                (float(score), row)
                for row in self.rows
                if (score := self._first_present(row, self.score_keys)) is not None
            ),
            key=lambda item: item[0],
        )
        if len(scored_rows) <= n:
            return [row for _, row in scored_rows]
        if n == 1:
            return [scored_rows[len(scored_rows) // 2][1]]

        scores = [score for score, _ in scored_rows]
        minimum = scores[0]
        score_step = (scores[-1] - minimum) / (n - 1)
        selected: List[Dict[str, Any]] = []
        used_indices = set()

        for sample_index in range(n):
            target = minimum + sample_index * score_step
            insertion = bisect_left(scores, target)
            candidates = [
                index
                for index in (insertion - 1, insertion)
                if 0 <= index < len(scored_rows) and index not in used_indices
            ]
            if candidates:
                chosen = min(candidates, key=lambda index: abs(scores[index] - target))
            else:
                chosen = next(index for index in range(len(scored_rows)) if index not in used_indices)
            used_indices.add(chosen)
            selected.append(scored_rows[chosen][1])

        random.Random(f"tad66k|{self.split}|score-spaced|{n}").shuffle(selected)
        return selected

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        rating = self.get_rating_from_row(row)
        return [str(rating)] if rating is not None else []

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("image_path"):
            return Image.open(Path(str(row["image_path"]))).convert("RGB")
        try:
            from remotezip import RemoteZip
        except ImportError as exc:
            raise ImportError("Streaming TAD66K requires the `remotezip` package.") from exc
        if self._remote_zip is None:
            self._remote_zip = RemoteZip(self.image_archive_url)
        image_bytes = self._remote_zip.read(str(row["image_name"]))
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def get_rating_from_row(self, row: Dict[str, Any]) -> int | None:
        value = row.get("rating")
        if value is None:
            value = self._first_present(row, self.score_keys)
        if value is None or str(value).strip() == "":
            return None
        return max(1, min(10, int(float(value) + 0.5)))

    def _load_remote_scores(self) -> List[Dict[str, Any]]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Streaming TAD66K requires `huggingface_hub`.") from exc
        labels_path = Path(
            hf_hub_download(
                self.dataset_id,
                "labels.zip",
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
            )
        )
        member = f"labels/merge/{self.split}.csv"
        with zipfile.ZipFile(labels_path) as archive:
            with archive.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                return [
                    {
                        "image_name": str(row["image"]).strip(),
                        "score": float(row["score"]),
                        "rating": self.get_rating_from_row(row),
                    }
                    for row in reader
                    if row.get("image") and row.get("score")
                ]

    def _find_metadata_file(self, metadata_file: str | os.PathLike[str] | None) -> Path:
        assert self.data_dir is not None
        if metadata_file:
            path = Path(metadata_file)
            if not path.is_absolute():
                path = self.data_dir / path
            if path.exists():
                return path
            raise FileNotFoundError(f"TAD66K metadata file not found: {path}")
        candidates = [
            *self.data_dir.glob("*.csv"),
            *self.data_dir.glob("*.txt"),
            *self.data_dir.glob("**/*score*.csv"),
            *self.data_dir.glob("**/*label*.csv"),
            *self.data_dir.glob("**/*score*.txt"),
            *self.data_dir.glob("**/*label*.txt"),
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"No TAD66K metadata CSV/TXT was found under {self.data_dir}.")

    def _load_local_rows(self) -> List[Dict[str, Any]]:
        assert self.metadata_path is not None
        rows: List[Dict[str, Any]] = []
        with self.metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            for raw in reader:
                split_value = str(raw.get("split", raw.get("set", ""))).strip().lower()
                if split_value and not split_value.startswith(self.split.lower()[:3]):
                    continue
                image_value = self._first_present(raw, self.image_keys)
                score_value = self._first_present(raw, self.score_keys)
                if image_value is None or score_value is None:
                    continue
                image_path = self._resolve_image_path(str(image_value))
                if image_path is None:
                    continue
                row = dict(raw)
                row["image_path"] = str(image_path)
                row["rating"] = self.get_rating_from_row(row)
                rows.append(row)
        if not rows:
            raise ValueError(f"No usable TAD66K rows were found in {self.metadata_path}.")
        return rows

    def _resolve_image_path(self, value: str) -> Path | None:
        assert self.data_dir is not None
        candidate = Path(value)
        paths = [candidate] if candidate.is_absolute() else [
            self.data_dir / candidate,
            self.data_dir / "images" / candidate,
            self.data_dir / "Image" / candidate,
        ]
        for path in paths:
            if path.exists():
                return path
        matches = list(self.data_dir.glob(f"**/{candidate.name}"))
        return matches[0] if matches else None

    @staticmethod
    def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None
