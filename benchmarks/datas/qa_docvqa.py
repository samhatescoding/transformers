from __future__ import annotations

from .hf_common import HFQADataset


class DocVQA(HFQADataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "lmms-lab/DocVQA") -> None:
        actual_split = "validation" if split.startswith("val") else split
        super().__init__(name="docvqa", dataset_id=dataset_id, config_name="DocVQA", split=actual_split, streaming=streaming)
