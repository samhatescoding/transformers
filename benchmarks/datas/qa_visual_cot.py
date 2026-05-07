from __future__ import annotations

from .hf_common import HFQADataset


class VisualCoT(HFQADataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "visual-cot/visual-cot") -> None:
        super().__init__(name="visual_cot", dataset_id=dataset_id, split=split, streaming=streaming)
