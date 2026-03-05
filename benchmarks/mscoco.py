from __future__ import annotations

from typing import Any, Dict, List, Set

from data import MSCOCO

from .base import BaseBenchmark


class MSCOCOBenchmark(BaseBenchmark):
    def __init__(self, split: str = "validation", streaming: bool = True):
        super().__init__(dataset=MSCOCO(split=split, streaming=streaming), name="mscoco")

    def _coerce_label(self, value: Any) -> str | None:
        if isinstance(value, int):
            if 0 <= value < len(self.dataset.labels):
                return self.dataset.labels[value]
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("label", "name", "category", "category_name", "class"):
                if key in value:
                    return self._coerce_label(value[key])
        return None

    def _flatten_labels(self, value: Any) -> List[str]:
        out: List[str] = []
        if isinstance(value, list):
            for item in value:
                out.extend(self._flatten_labels(item))
            return out
        coerced = self._coerce_label(value)
        return [coerced] if coerced else []

    def _labels_from_row_annotations(self, row: Dict[str, Any]) -> List[str]:
        labels: Set[str] = set()
        containers = []
        for key in ("objects", "annotations", "instances", "segments_info"):
            if key in row and row[key] is not None:
                containers.append(row[key])

        for container in containers:
            if isinstance(container, dict):
                for key in (
                    "category",
                    "categories",
                    "category_id",
                    "category_ids",
                    "label",
                    "labels",
                    "name",
                    "names",
                    "class",
                    "classes",
                ):
                    if key in container:
                        labels.update(self._flatten_labels(container[key]))
            else:
                labels.update(self._flatten_labels(container))

        return sorted(labels)

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        labels = self._labels_from_row_annotations(row)
        if labels:
            return labels
        return super().get_valid_labels_for_row(row)

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        labels: Set[str] = set()
        for row in rows:
            labels.update(self._labels_from_row_annotations(row))
        if labels:
            return sorted(labels)
        return super().get_candidate_labels(rows)
