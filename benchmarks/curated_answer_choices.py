from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CHOICES_ROOT = Path(__file__).resolve().parents[1] / "benchmark_choices" / "type_a"


@lru_cache(maxsize=None)
def _load_dataset(dataset_name: str) -> tuple[dict[str, Any], ...]:
    path = CHOICES_ROOT / f"{dataset_name}.json"
    if not path.exists():
        return ()

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(payload.get("rows", ()))
    for expected_index, row in enumerate(rows):
        if row.get("row_index") != expected_index:
            raise ValueError(f"{path} row indexes must be contiguous from zero.")
        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"{path} row {expected_index} must contain at least two choices.")
        normalized = {str(choice).strip().casefold() for choice in choices}
        if "" in normalized or len(normalized) != len(choices):
            raise ValueError(f"{path} row {expected_index} contains blank or duplicate choices.")
        annotation_status = row.get("annotation_status", "ready")
        if annotation_status == "draft":
            continue
        if annotation_status != "ready":
            raise ValueError(
                f"{path} row {expected_index} has unknown annotation_status "
                f"{annotation_status!r}."
            )
        answer = str(row.get("answer", "")).strip()
        answer_matches = [
            index
            for index, choice in enumerate(choices)
            if str(choice).strip().casefold() == answer.casefold()
        ]
        if len(answer_matches) != 1:
            raise ValueError(
                f"{path} row {expected_index} must contain its answer exactly once."
            )
        if row.get("correct_choice_index") != answer_matches[0]:
            raise ValueError(
                f"{path} row {expected_index} has an incorrect correct_choice_index."
            )
    return rows


def get_curated_answer_row(dataset_name: str, row_index: int) -> dict[str, Any] | None:
    rows = _load_dataset(str(dataset_name))
    if row_index < 0 or row_index >= len(rows):
        return None
    row = rows[row_index]
    if row.get("annotation_status", "ready") != "ready":
        return None
    return dict(row)
