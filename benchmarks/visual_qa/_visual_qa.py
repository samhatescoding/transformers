from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple

from ..curated_answer_choices import get_curated_answer_row
from .._base_benchmark import BaseBenchmark


class VisualQABenchmark(BaseBenchmark):
    def prepare(self, n: int, label_sample_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        rows = self.dataset.get_samples(n)
        prepared_rows = []
        for row_index, source_row in enumerate(rows):
            row = dict(source_row)
            curated = get_curated_answer_row(self.dataset.name, row_index)
            if curated is not None:
                try:
                    self._validate_curated_row(row=row, curated=curated, row_index=row_index)
                except ValueError:
                    curated = None
            if curated is not None:
                row["choices"] = list(curated["choices"])
                row["answer"] = str(curated["answer"])
                row["curated_choice_row_index"] = row_index
            prepared_rows.append(row)
        return prepared_rows, []

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        del rows
        return []

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        del labels
        return [str(choice) for choice in row.get("choices", [])]

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del image
        if row is None:
            raise ValueError("VisualQABenchmark requires a dataset row.")
        question = self._get_question(row)
        choices = labels or [str(choice) for choice in row.get("choices", [])]
        if choices:
            rendered_choices = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
            return (
                "USER: <image>\n"
                "Answer the question about the image.\n"
                f"Question: {question}\n"
                "Choices:\n"
                f"{rendered_choices}\n"
                "Return either the choice letter or the exact choice text.\n"
                "ASSISTANT:"
            )
        return (
            "USER: <image>\n"
            "Answer the question about the image.\n"
            "Return only the answer text.\n"
            f"Question: {question}\n"
            "ASSISTANT:"
        )

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        return list(self._get_answers(row))

    def evaluate_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        image: Any | None = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        del image
        answers = self._get_answers(row)
        normalized_answers = sorted({self.dataset.normalize_text(answer) for answer in answers if str(answer).strip()})
        choices = [str(choice) for choice in row.get("choices", [])]
        selected = self._parse_choice(prediction, choices) if choices else str(prediction)
        pred_norm = self.dataset.normalize_text(selected)
        return (
            pred_norm in set(normalized_answers),
            normalized_answers,
            {
                "reference_answers": answers,
                "choices": choices,
                "selected_choice": selected,
            },
        )

    def _get_question(self, row: Dict[str, Any]) -> str:
        getter = getattr(self.dataset, "get_question_from_row", None)
        if callable(getter):
            return str(getter(row))
        return str(row.get("question", "")).strip()

    def _get_answers(self, row: Dict[str, Any]) -> Sequence[str]:
        if row.get("answer") is not None and row.get("choices"):
            return [str(row["answer"])]
        getter = getattr(self.dataset, "get_answers_from_row", None)
        if callable(getter):
            return list(getter(row))
        answers = row.get("answers")
        if isinstance(answers, list):
            return [str(item) for item in answers]
        answer = row.get("answer")
        if answer is None:
            return []
        return [str(answer)]

    def _validate_curated_row(
        self,
        *,
        row: Dict[str, Any],
        curated: Dict[str, Any],
        row_index: int,
    ) -> None:
        source_question = self._comparison_text(self._get_question(row))
        curated_question = self._comparison_text(curated.get("question", ""))
        if source_question != curated_question:
            raise ValueError(
                f"Curated choices for {self.dataset.name} row {row_index} do not match "
                f"the dataset question: {curated.get('question')!r} != {self._get_question(row)!r}"
            )

        source_answers = {
            self.dataset.normalize_text(answer)
            for answer in self._get_answers(row)
            if str(answer).strip()
        }
        curated_answer = self.dataset.normalize_text(curated.get("answer", ""))
        if curated_answer not in source_answers:
            raise ValueError(
                f"Curated answer for {self.dataset.name} row {row_index} is not one "
                f"of the dataset answers: {curated.get('answer')!r}"
            )

    @staticmethod
    def _comparison_text(value: Any) -> str:
        ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
        return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))

    def _parse_choice(self, prediction: str, choices: Sequence[str]) -> str:
        pred = str(prediction or "").strip()
        pred_norm = self.dataset.normalize_text(pred)
        for choice in choices:
            if pred_norm == self.dataset.normalize_text(choice):
                return str(choice)
        match = re.match(r"^\s*([A-Z])\b", pred, re.IGNORECASE)
        if match:
            index = ord(match.group(1).upper()) - 65
            if 0 <= index < len(choices):
                return str(choices[index])
        return pred

    def analyze_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        prompt_labels: List[str],
        valid_labels: List[str],
        predicted_boxes: List[Dict[str, Any]],
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        del row
        del prompt_labels
        del valid_labels
        del predicted_boxes
        del evaluation
        stripped = str(prediction or "").strip()
        return {
            "generated_output_count": 0 if not stripped else 1,
            "hallucinated_label_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "predicted_detection_count": 0,
        }
