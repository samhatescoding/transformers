# datasets/mscoco.py

from __future__ import annotations
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from io import BytesIO
from urllib.request import urlopen

from PIL import Image

from ._base_dataset import BaseDataset


class MSCOCO(BaseDataset):
    COCO_CATEGORY_ID_TO_NAME = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }

    def __init__(
        self,
        split: str = "validation",
        streaming: bool = True,
        dataset_id: str = "phiyodr/coco2017",
        annotations_file: str | None = None,
        auto_download_annotations: bool = True,
    ):
        self.name = "mscoco"
        self.split = split
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.annotations_file = annotations_file
        self.auto_download_annotations = auto_download_annotations

        from datasets import load_dataset

        self.ds = load_dataset(
            self.dataset_id,
            split=split,
            streaming=streaming,
        )

        # Official COCO 80 detection/instance categories (some multi-word)
        self.labels = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        ]
        self.category_id_to_label = dict(self.COCO_CATEGORY_ID_TO_NAME)
        self._annotations_by_image_id: Dict[int, List[Dict[str, Any]]] | None = None

    def __repr__(self) -> str:
        return f"<Dataset {self.name} | repo={self.dataset_id} | split={self.split} | streaming={self.streaming}>"

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.ds)
    
    def get_labels_img(self, row) -> List[str]:
        labels = set()

        # First prefer labels from COCO official annotations file if available.
        for ann in self.get_annotations_for_row(row):
            label = ann.get("label")
            if isinstance(label, str) and label.strip():
                labels.add(label.strip())

        # Fallback to any labels embedded in the row payload.
        if not labels:
            self._collect_labels_from_container(row.get("objects"), labels)
            self._collect_labels_from_container(row.get("annotations"), labels)
            self._collect_labels_from_container(row.get("instances"), labels)
            self._collect_labels_from_container(row.get("segments_info"), labels)

        return sorted(labels)
    
    def get_labels(self, rows) -> List[str]:
        return self.labels

    def get_annotations_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._ensure_annotations_loaded()
        if not self._annotations_by_image_id:
            return []

        image_id = self._extract_image_id(row)
        if image_id is None:
            return []

        out: List[Dict[str, Any]] = []
        for ann in self._annotations_by_image_id.get(image_id, []):
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            category_id = ann.get("category_id")
            label = self._coerce_label(category_id) or ""
            out.append({"bbox": bbox, "label": label, "category_id": category_id})
        return out

    def _coerce_label(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, int):
            if value in self.category_id_to_label:
                return self.category_id_to_label[value]
            if 0 <= value < len(self.labels):
                return self.labels[value]
        return None

    def _collect_labels_from_container(self, container: Any, labels: set[str]) -> None:
        if container is None:
            return
        if isinstance(container, list):
            for item in container:
                self._collect_labels_from_container(item, labels)
            return
        if isinstance(container, dict):
            for key in ("category", "category_id", "categories", "category_ids", "label", "labels", "name", "names", "class", "classes"):
                if key in container:
                    self._collect_labels_from_container(container[key], labels)
            return

        label = self._coerce_label(container)
        if label:
            labels.add(label)

    def _extract_image_id(self, row: Dict[str, Any]) -> Optional[int]:
        for key in ("image_id", "id"):
            if key in row:
                raw = row[key]
                if isinstance(raw, int):
                    return raw
                if isinstance(raw, str) and raw.isdigit():
                    return int(raw)

        url = self._get_url(row)
        if not url:
            return None

        filename = url.rstrip("/").split("/")[-1]
        stem = filename.split(".")[0]
        if stem.isdigit():
            return int(stem)
        return None

    def _ensure_annotations_loaded(self) -> None:
        if self._annotations_by_image_id is not None:
            return

        annotation_path = self._resolve_annotations_path()
        if annotation_path is None or not annotation_path.exists():
            self._annotations_by_image_id = {}
            return

        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception:
            self._annotations_by_image_id = {}
            return

        annotations = payload.get("annotations", [])
        categories = payload.get("categories", [])
        if isinstance(categories, list):
            for cat in categories:
                if isinstance(cat, dict):
                    cid = cat.get("id")
                    name = cat.get("name")
                    if isinstance(cid, int) and isinstance(name, str) and name.strip():
                        self.category_id_to_label[cid] = name.strip()

        by_image_id: Dict[int, List[Dict[str, Any]]] = {}
        if isinstance(annotations, list):
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                image_id = ann.get("image_id")
                if not isinstance(image_id, int):
                    continue
                by_image_id.setdefault(image_id, []).append(ann)

        self._annotations_by_image_id = by_image_id

    def _resolve_annotations_path(self) -> Optional[Path]:
        if self.annotations_file:
            path = Path(self.annotations_file)
            return path if path.exists() else None

        split_key = "val2017" if self.split.startswith("val") or self.split.startswith("validation") else "train2017"
        target_name = f"instances_{split_key}.json"
        base_dir = Path(".tmp") / "coco_annotations"
        target_path = base_dir / target_name
        if target_path.exists():
            return target_path

        if not self.auto_download_annotations:
            return None

        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            zip_path = base_dir / "annotations_trainval2017.zip"
            if not zip_path.exists():
                with urlopen("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", timeout=120) as resp:
                    zip_path.write_bytes(resp.read())

            with zipfile.ZipFile(zip_path, "r") as zf:
                member = f"annotations/{target_name}"
                with zf.open(member) as src:
                    target_path.write_bytes(src.read())
        except Exception:
            return None

        return target_path if target_path.exists() else None

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(self.ds):
            if i >= n:
                break
            samples.append(row)
        return samples

    def _get_url(self, row: Dict[str, Any]) -> Optional[str]:
        return row.get("coco_url") or row.get("image_url") or row.get("url")

    def get_image_from_row(self, row: Dict[str, Any]) -> Image.Image:
        img = row.get("image")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                return Image.open(BytesIO(img["bytes"])).convert("RGB")
            if "path" in img and img["path"] is not None:
                return Image.open(img["path"]).convert("RGB")

        url = self._get_url(row)
        if not url:
            raise ValueError(
                "No image data found in row. Expected one of: "
                "row['image'] or coco_url/image_url/url."
            )

        with urlopen(url, timeout=30) as r:
            return Image.open(BytesIO(r.read())).convert("RGB")

    def get_url_from_row(self, row: Dict[str, Any]) -> str:
        url = self._get_url(row)
        if not url:
            raise ValueError("No image URL found in dataset row (expected coco_url/image_url/url).")
        return url
