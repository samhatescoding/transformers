from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
from datasets import load_dataset

from .base import BaseDataset


class INaturalist(BaseDataset):
    """
    iNaturalist dataset adapter for image workflows.
    Supports:
      - Standard classification datasets (label/category/species fields)
      - Nested taxonomy-style records
      - Chat-style datasets (e.g. sxj1215/inaturalist)
    """

    def __init__(
        self,
        split: str = "train",
        streaming: bool = True,
        dataset_id: Optional[str] = "sxj1215/inaturalist",
        config_name: Optional[str] = None,
        use_auth_token: Optional[bool] = None,
    ):
        self.name = "inaturalist"
        self.split = split
        self.streaming = streaming

        if not dataset_id:
            raise ValueError(
                "INaturalist requires an explicit Hugging Face dataset repo id. "
                "Example: INaturalist(dataset_id='owner/repo', split='train')."
            )

        self.dataset_id = dataset_id
        self.config_name = config_name

        load_kwargs: Dict[str, Any] = {
            "split": split,
            "streaming": streaming,
            "use_auth_token": use_auth_token,
        }

        if config_name:
            self.ds = load_dataset(dataset_id, config_name, **load_kwargs)
        else:
            self.ds = load_dataset(dataset_id, **load_kwargs)

        self.class_labels: List[str] = self._extract_class_labels()
        self.default_label = self._infer_default_label()
        self.labels = list(self.class_labels)

        if not self.labels:
            inferred = self._bootstrap_labels_from_rows(sample_size=256)
            if inferred:
                self.labels = inferred

        if not self.labels and self.default_label:
            self.labels = [self.default_label]

    def __repr__(self) -> str:
        return f"<Dataset {self.name} | split={self.split} | streaming={self.streaming}>"

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.ds)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(self.ds):
            if i >= n:
                break
            samples.append(row)
        return samples

    # ---------------------------------------------------------------------
    # Label Extraction
    # ---------------------------------------------------------------------

    def _extract_class_labels(self) -> List[str]:
        try:
            for key in ("label", "class", "category_id", "species_id"):
                if key in self.ds.features:
                    feature = self.ds.features[key]
                    names = list(getattr(feature, "names", []))
                    if names:
                        return names
        except Exception:
            pass
        return []

    @staticmethod
    def _path_key(path: str) -> str:
        last = path.split(".")[-1]
        if "[" in last:
            last = last.split("[", 1)[0]
        return last.strip().lower()

    @staticmethod
    def _is_meaningful_label(text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        if t.lower() == "inaturalist":
            return False
        return True

    @staticmethod
    def _split_label_text(text: str) -> List[str]:
        if not text:
            return []
        t = str(text).strip()
        if not t:
            return []

        if ":" in t and len(t.split(":", 1)[0]) <= 30:
            head, tail = t.split(":", 1)
            if head.strip().lower() in {
                "species",
                "scientific name",
                "scientific_name",
                "taxon",
                "label",
                "name",
                "class",
                "category",
            }:
                t = tail.strip()

        parts: List[str] = []
        for chunk in t.replace("\r", "\n").split("\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            for sub in chunk.replace(";", ",").split(","):
                s = sub.strip().strip('"').strip("'")
                if s:
                    parts.append(s)

        return parts or [t]

    def _labels_from_messages(self, row: Dict[str, Any]) -> List[str]:
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            return []

        out: List[str] = []
        seen = set()

        def add_many(raw_text: Any) -> None:
            for cand in self._split_label_text(str(raw_text)):
                if not self._is_meaningful_label(cand):
                    continue
                n = self.normalize_text(cand)
                if n in seen:
                    continue
                seen.add(n)
                out.append(cand)

        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).lower().strip()
            content = m.get("content")
            if role == "assistant" and content:
                add_many(content)

        if not out:
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("content"):
                    add_many(m["content"])
                    if out:
                        break

        return out

    def _labels_from_row(self, row: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        seen = set()

        def add_label(raw: Any) -> None:
            if raw is None:
                return
            if isinstance(raw, int) and 0 <= raw < len(self.class_labels):
                label = self.class_labels[raw]
            else:
                label = str(raw).strip()
            if not self._is_meaningful_label(label):
                return
            nlabel = self.normalize_text(label)
            if nlabel in seen:
                return
            seen.add(nlabel)
            out.append(label)

        # Chat-style datasets first
        for ml in self._labels_from_messages(row):
            add_label(ml)

        for key in ("label", "class", "category", "species", "taxon_name", "scientific_name"):
            if row.get(key) is not None:
                add_label(row[key])

        for key in ("category_id", "species_id"):
            if row.get(key) is not None:
                add_label(row[key])

        nested_key_hints = (
            "scientific_name",
            "taxon",
            "species",
            "common_name",
            "category",
            "label",
            "class",
            "name",
        )

        for path, value in self._iter_nested_items(row):
            key = self._path_key(path)
            if not any(hint in key for hint in nested_key_hints):
                continue
            if isinstance(value, (str, int)):
                add_label(value)

        if not out:
            captions = self._captions_from_row(row)
            if captions:
                for noun in sorted(self.extract_nouns(captions)):
                    add_label(noun)

        return out

    def _label_from_row(self, row: Dict[str, Any]) -> Optional[str]:
        labels = self._labels_from_row(row)
        if labels:
            return labels[0]
        return self.default_label

    def get_labels_img(self, row: Dict[str, Any]) -> List[str]:
        labels = self._labels_from_row(row)
        if labels:
            return labels
        if self.class_labels:
            return list(self.class_labels)
        if self.default_label:
            return [self.default_label]
        return []

    def get_labels(
        self,
        rows,
        *,
        min_unique: int = 32,
        max_scan: int = 1000,
    ) -> List[str]:
        """
        Return a set of candidate labels for prompting.

        For streaming datasets, the first N rows can be highly non-diverse (often grouped).
        So if we don't collect enough unique labels from `rows`, keep scanning the dataset
        until we reach `min_unique` unique labels or we scan `max_scan` rows total.

        - rows: iterable of rows already sampled (e.g., get_samples(64))
        - min_unique: target number of unique labels to return (best-effort)
        - max_scan: maximum rows to scan from the underlying dataset (best-effort)
        """
        merged: List[str] = []
        seen = set()

        def add_label(label: str) -> None:
            nlabel = self.normalize_text(label)
            if nlabel in seen:
                return
            seen.add(nlabel)
            merged.append(label)

        # 1) First, use the provided rows
        provided_rows: List[Dict[str, Any]] = list(rows) if rows is not None else []
        for row in provided_rows:
            for label in self.get_labels_img(row):
                add_label(label)

        # 2) If still not enough, scan more of the dataset
        #    (this is the key fix for your current behavior)
        scanned = 0
        if len(merged) < min_unique:
            try:
                for row in self.ds:
                    scanned += 1
                    for label in self.get_labels_img(row):
                        add_label(label)
                        if len(merged) >= min_unique:
                            break
                    if len(merged) >= min_unique or scanned >= max_scan:
                        break
            except Exception:
                pass

        # 3) Fallbacks
        if not merged:
            existing = [l for l in self.labels if self._is_meaningful_label(l)]
            if existing:
                merged = existing

        if not merged and self.class_labels:
            merged = list(self.class_labels)

        if not merged and self.default_label:
            merged = [self.default_label]

        self.labels = merged
        return merged

    # ---------------------------------------------------------------------
    # Bootstrap / Captions
    # ---------------------------------------------------------------------

    def _bootstrap_labels_from_rows(self, sample_size: int = 256) -> List[str]:
        merged: List[str] = []
        seen = set()

        try:
            for i, row in enumerate(self.ds):
                if i >= sample_size:
                    break
                for label in self._labels_from_row(row):
                    nlabel = self.normalize_text(label)
                    if nlabel in seen:
                        continue
                    seen.add(nlabel)
                    merged.append(label)
        except Exception:
            return []

        return merged

    def _captions_from_row(self, row: Dict[str, Any]) -> List[str]:
        direct_keys = (
            "caption",
            "captions",
            "text",
            "description",
            "title",
            "summary",
            "notes",
            "observation",
        )

        out: List[str] = []

        for key in direct_keys:
            if key in row and isinstance(row[key], str):
                out.append(row[key])

        msgs = row.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    content = m.get("content")
                    role = str(m.get("role", "")).lower()
                    if role in ("user", "system") and isinstance(content, str):
                        out.append(content)

        return out

    # ---------------------------------------------------------------------
    # Image Extraction
    # ---------------------------------------------------------------------

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        image_key_order = ("image", "img", "rgb", "photo", "pixel_values")

        for key in image_key_order:
            if key in row and row[key] is not None:
                return self._image_from_any(row[key]).convert("RGB")

        for _, value in self._iter_nested_items(row):
            if value is not None:
                try:
                    return self._image_from_any(value).convert("RGB")
                except Exception:
                    pass

        raise ValueError(f"Could not extract image. Row keys: {sorted(row.keys())}")

    def _infer_default_label(self) -> Optional[str]:
        if self.config_name:
            return self._prettify_label(self.config_name)
        tail = self.dataset_id.split("/")[-1]
        if tail:
            return self._prettify_label(tail)
        return None

    @staticmethod
    def _prettify_label(text: str) -> str:
        return text.replace("_", " ").replace("-", " ").strip()

    @staticmethod
    def _image_from_any(obj: Any) -> Image.Image:
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, dict):
            if "bytes" in obj and obj["bytes"] is not None:
                return Image.open(BytesIO(obj["bytes"]))
            if "path" in obj and obj["path"]:
                return Image.open(obj["path"])
        if isinstance(obj, str):
            return Image.open(obj)
        if isinstance(obj, (list, tuple)) and obj:
            return INaturalist._image_from_any(obj[0])
        return Image.fromarray(obj)

    @staticmethod
    def _iter_nested_items(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                yield path, v
                yield from INaturalist._iter_nested_items(v, path)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:3]):
                path = f"{prefix}[{i}]"
                yield path, v
                yield from INaturalist._iter_nested_items(v, path)