from __future__ import annotations

from typing import Any, Dict, List, Set

from data import Flickr30k

from .base import BaseBenchmark


class Flickr30kBenchmark(BaseBenchmark):
    def __init__(self, split: str = "test", streaming: bool = True):
        super().__init__(dataset=Flickr30k(split=split, streaming=streaming), name="flickr30k")

    def _collect_text_labels(self, value: Any, out: Set[str]) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                out.add(text)
            return
        if isinstance(value, list):
            for item in value:
                self._collect_text_labels(item, out)
            return
        if isinstance(value, dict):
            for key in ("label", "labels", "phrase", "phrases", "entity", "entities", "name", "names"):
                if key in value:
                    self._collect_text_labels(value[key], out)

    def _labels_from_regions(self, row: Dict[str, Any]) -> List[str]:
        labels: Set[str] = set()

        # Common Flickr30k entities/regions variants.
        for key in ("annotations", "regions", "boxes", "bbox", "bboxes", "rectangles", "entities"):
            if key in row and row[key] is not None:
                self._collect_text_labels(row[key], labels)

        return sorted(labels)

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        labels = self._labels_from_regions(row)
        if labels:
            return labels
        return super().get_valid_labels_for_row(row)

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        labels: Set[str] = set()
        for row in rows:
            labels.update(self._labels_from_regions(row))
        if labels:
            return sorted(labels)
        return super().get_candidate_labels(rows)
