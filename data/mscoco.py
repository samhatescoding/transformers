# datasets/mscoco.py

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
from io import BytesIO
from urllib.request import urlopen

from PIL import Image

from .base import BaseDataset


class MSCOCO(BaseDataset):
    def __init__(self, split: str = "validation", streaming: bool = True):
        self.name = "mscoco"
        self.split = split
        self.streaming = streaming

        from datasets import load_dataset

        self.ds = load_dataset(
            "phiyodr/coco2017",
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

    def __repr__(self) -> str:
        return f"<Dataset {self.name} | split={self.split} | streaming={self.streaming}>"

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        return iter(self.ds)
    
    def get_labels_img(self, row) -> List[str]:
        return self.labels
    
    def get_labels(self, rows) -> List[str]:
        return self.labels

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
        url = self._get_url(row)
        if not url:
            raise ValueError("No image URL found in dataset row (expected coco_url/image_url/url).")

        with urlopen(url, timeout=30) as r:
            return Image.open(BytesIO(r.read())).convert("RGB")

    def get_url_from_row(self, row: Dict[str, Any]) -> str:
        url = self._get_url(row)
        if not url:
            raise ValueError("No image URL found in dataset row (expected coco_url/image_url/url).")
        return url
