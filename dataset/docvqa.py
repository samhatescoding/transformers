from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .hf_common import HFQADataset


class DocVQA(HFQADataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "lmms-lab/DocVQA") -> None:
        actual_split = "validation" if split.startswith("val") else split
        super().__init__(name="docvqa", dataset_id=dataset_id, config_name="DocVQA", split=actual_split, streaming=streaming)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        seen_doc_ids = set()
        for index, row in enumerate(self.ds):
            if index == 0:
                continue
            doc_id = str(row.get("docId", "")).strip()
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            yield self._standardize_row(row)

    def get_samples(self, n: int) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        seen_doc_ids = set()
        for index, row in enumerate(self.ds):
            if index == 0:
                continue
            doc_id = str(row.get("docId", "")).strip()
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            samples.append(self._standardize_row(row))
            if len(samples) >= n:
                break
        return samples
