from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence


CURATION_ROOT = Path(__file__).resolve().parents[1] / "benchmark_choices"
MANUAL_ROOTS = (
    CURATION_ROOT / "type_e",
    CURATION_ROOT / "type_g",
    CURATION_ROOT / "manual",
)


@lru_cache(maxsize=1)
def _dataset_banks() -> dict[str, str]:
    path = CURATION_ROOT / "datasets.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load_bank(bank_name: str) -> tuple[dict[str, object], ...]:
    path = CURATION_ROOT / f"{bank_name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(payload)
    if len(entries) < 20:
        raise ValueError(f"{path} must contain at least 20 curated attribute categories.")
    for entry in entries:
        replacements = entry.get("replacements", {})
        if not entry.get("category") or not isinstance(replacements, dict):
            raise ValueError(f"Invalid attribute entry in {path}: {entry!r}")
        if any(not source or not isinstance(options, list) or len(options) < 2 for source, options in replacements.items()):
            raise ValueError(f"Every source phrase in {path} needs at least two replacements.")
    return entries


def _replace_once(text: str, source: str, replacement: str) -> str:
    pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
    return pattern.sub(replacement, text, count=1)


@lru_cache(maxsize=None)
def _load_manual_dataset(dataset_name: str) -> tuple[dict[str, object], ...]:
    filename = f"{dataset_name}.json"
    for root in MANUAL_ROOTS:
        path = root / filename
        if path.exists():
            break
    else:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(payload.get("rows", ()))
    for expected_index, row in enumerate(rows):
        if row.get("row_index") != expected_index:
            raise ValueError(f"{path} row indexes must be contiguous from zero.")
        distractors = row.get("distractors")
        if not isinstance(distractors, list) or len(distractors) != 3:
            raise ValueError(f"{path} row {expected_index} must contain exactly three distractors.")
    return rows


def get_curated_distractors(
    dataset_name: str,
    row_index: int,
    correct_prompt: str,
) -> Sequence[str]:
    manual_rows = _load_manual_dataset(str(dataset_name))
    if row_index < len(manual_rows):
        row = manual_rows[row_index]
        expected_prompt = str(row.get("correct_prompt", "")).strip()
        if expected_prompt == str(correct_prompt).strip():
            distractors = tuple(str(item).strip() for item in row["distractors"])
            if not all(distractors) or len({expected_prompt, *distractors}) != 4:
                raise ValueError(f"Invalid manual distractors for {dataset_name} row {row_index}.")
            return distractors

    bank_name = _dataset_banks().get(str(dataset_name))
    if bank_name is None or row_index >= 20:
        return ()
    bank = _load_bank(bank_name)
    rotated = [*bank[row_index % len(bank):], *bank[:row_index % len(bank)]]
    matches: list[tuple[str, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for entry in rotated:
        replacements = entry["replacements"]
        for source in sorted(replacements, key=len, reverse=True):
            match = re.search(rf"(?<!\w){re.escape(source)}(?!\w)", correct_prompt, re.IGNORECASE)
            if match is None or any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            options = replacements[source]
            replacement = str(options[row_index % len(options)])
            matches.append((source, replacement, str(entry["category"])))
            occupied.append(match.span())
            break
        if len(matches) == 2:
            break

    if len(matches) != 2:
        raise ValueError(
            f"Could not find two curated attribute substitutions for row {row_index} "
            f"of {dataset_name}: {correct_prompt!r}"
        )

    source_a, replacement_a, _ = matches[0]
    source_b, replacement_b, _ = matches[1]
    changed_a = _replace_once(correct_prompt, source_a, replacement_a)
    changed_b = _replace_once(correct_prompt, source_b, replacement_b)
    changed_both = _replace_once(changed_a, source_b, replacement_b)
    distractors = (changed_a, changed_b, changed_both)
    if len({correct_prompt, *distractors}) != 4:
        raise ValueError(f"Balanced substitutions did not produce four unique prompts for {dataset_name} row {row_index}.")
    return distractors
