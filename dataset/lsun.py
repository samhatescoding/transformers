from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

from ._base_dataset import BaseDataset


class LSUN(BaseDataset):
    """LSUN scene classification from local images or cached HF validation LMDBs."""

    SCENE_CATEGORIES = (
        "bedroom",
        "bridge",
        "church outdoor",
        "classroom",
        "conference room",
        "dining room",
        "kitchen",
        "living room",
        "restaurant",
        "tower",
    )
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    DEFAULT_DATASET_ID = "RichardErkhov/LSUN"

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        data_dir: str | os.PathLike[str] | None = None,
        dataset_id: str = DEFAULT_DATASET_ID,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        del streaming
        self.name = "lsun"
        self.split = split
        self.labels = list(self.SCENE_CATEGORIES)
        self.dataset_id = dataset_id
        root = data_dir or os.environ.get("LSUN_ROOT")
        self.data_dir = Path(root).expanduser() if root else None
        if self.data_dir is not None and not self.data_dir.exists():
            raise FileNotFoundError(f"LSUN data directory not found: {self.data_dir}")
        default_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache / "lsun_validation_lmdb"
        self._local_rows: List[Dict[str, Any]] | None = None

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if self.data_dir is not None:
            return iter(self._get_local_rows())
        return self._iter_hf_rows()

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(self):
            if index >= n:
                break
            rows.append(row)
        return rows

    def get_labels(self, rows) -> List[str]:
        del rows
        return list(self.labels)

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        label = str(row.get("label", "")).strip()
        return [label] if label else []

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("image_bytes") is not None:
            return Image.open(BytesIO(row["image_bytes"])).convert("RGB")
        return Image.open(row["image_path"]).convert("RGB")

    def _get_local_rows(self) -> List[Dict[str, Any]]:
        if self._local_rows is None:
            self._local_rows = self._index_local_rows()
        return self._local_rows

    def _index_local_rows(self) -> List[Dict[str, Any]]:
        assert self.data_dir is not None
        rows: List[Dict[str, Any]] = []
        for label in self.SCENE_CATEGORIES:
            folder_names = {
                label,
                label.replace(" ", "_"),
                f"{label.replace(' ', '_')}_{self.split}",
            }
            category_paths = [self.data_dir / name for name in folder_names]
            category_paths += [self.data_dir / self.split / name for name in folder_names]
            category_path = next((path for path in category_paths if path.is_dir()), None)
            if category_path is None:
                continue
            for path in sorted(category_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in self.EXTENSIONS:
                    rows.append({"image_path": str(path), "label": label})
        if not rows:
            raise ValueError(
                f"No LSUN images were found under {self.data_dir}. Expected category directories such as "
                f"'bedroom_train' or 'train/bedroom'."
            )
        return rows

    def _iter_hf_rows(self) -> Iterable[Dict[str, Any]]:
        try:
            import lmdb
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Streaming LSUN requires `lmdb` and `huggingface_hub`.") from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for label in self.SCENE_CATEGORIES:
            slug = label.replace(" ", "_")
            filename = f"scenes/{slug}_val_lmdb.zip"
            archive_path = Path(
                hf_hub_download(
                    self.dataset_id,
                    filename,
                    repo_type="dataset",
                    token=os.getenv("HF_TOKEN"),
                )
            )
            lmdb_path = self.cache_dir / f"{slug}_val_lmdb"
            if not (lmdb_path / "data.mdb").exists():
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(self.cache_dir)

            environment = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=1,
            )
            try:
                with environment.begin() as transaction:
                    for _, image_bytes in transaction.cursor():
                        yield {"image_bytes": bytes(image_bytes), "label": label}
            finally:
                environment.close()
