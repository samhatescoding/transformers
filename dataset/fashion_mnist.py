from __future__ import annotations

from .hf_common import HFClassificationDataset


class FashionMNIST(HFClassificationDataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "zalando-datasets/fashion_mnist") -> None:
        super().__init__(name="fashion_mnist", dataset_id=dataset_id, split=split, streaming=streaming)
