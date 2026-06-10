from __future__ import annotations

import json
from pathlib import Path


CHOICES_PATH = (
    Path(__file__).resolve().parent / "type_a" / "flyingthings3d.json"
)


def main() -> None:
    payload = json.loads(CHOICES_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 FlyingThings3D rows, found {len(rows)}.")

    for expected_index, row in enumerate(rows):
        if row.get("row_index") != expected_index:
            raise ValueError(f"Row {expected_index} has a mismatched row_index.")

        question = str(row.get("question", "")).strip()
        question_template = str(row.get("question_template", "")).strip()
        if not question:
            if not question_template:
                raise ValueError(f"Row {expected_index} has no question.")
            row["question"] = question_template

        answer = str(row.get("answer", "")).strip()
        choices = row.get("choices")
        if not answer:
            raise ValueError(f"Row {expected_index} has no answer.")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"Row {expected_index} has invalid choices.")

        answer_matches = [
            index
            for index, choice in enumerate(choices)
            if str(choice).strip().casefold() == answer.casefold()
        ]
        if len(answer_matches) != 1:
            raise ValueError(
                f"Row {expected_index} must contain its answer exactly once."
            )

        row["correct_choice_index"] = answer_matches[0]
        row["annotation_status"] = "ready"

    CHOICES_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print("Finalized 20 FlyingThings3D annotations.")


if __name__ == "__main__":
    main()
