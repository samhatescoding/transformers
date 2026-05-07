from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .._base_benchmark import BaseBenchmark


class VisualQABenchmark(BaseBenchmark):
    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        del rows
        return []

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        del row
        del labels
        return []

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del labels
        del image
        if row is None:
            raise ValueError("VisualQABenchmark requires a dataset row.")
        question = self._get_question(row)
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
        pred_norm = self.dataset.normalize_text(prediction)
        return pred_norm in set(normalized_answers), normalized_answers, {"reference_answers": answers}

    def _get_question(self, row: Dict[str, Any]) -> str:
        getter = getattr(self.dataset, "get_question_from_row", None)
        if callable(getter):
            return str(getter(row))
        return str(row.get("question", "")).strip()

    def _get_answers(self, row: Dict[str, Any]) -> Sequence[str]:
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
