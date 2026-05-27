from __future__ import annotations

import re

from .hf_common import HFClassificationDataset


class FashionMNIST(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "zalando-datasets/fashion_mnist") -> None:
        super().__init__(name="fashion_mnist", dataset_id=dataset_id, split=split, streaming=streaming)

    def normalize_text(self, text: str) -> str:
        normalized = super().normalize_text(text)
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        return re.sub(r"\s*/\s*", "/", normalized)
