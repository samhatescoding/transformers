from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from .._base_benchmark import BaseBenchmark


class MultipleChoiceBenchmark(BaseBenchmark):
    default_instruction = "Choose the single best answer."
    fallback_distractors: Sequence[str] = ()
    fixed_choices: Sequence[str] | None = None

    def prepare(self, n: int, label_sample_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        rows_for_labels = self.dataset.get_samples(max(n, label_sample_size))
        rows = [self._ensure_choices(dict(row), rows_for_labels) for row in rows_for_labels[:n]]
        return rows, []

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        del rows
        return []

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        del labels
        return list(self._get_choices(row))

    def get_image_for_row(self, row: Dict[str, Any]) -> Image.Image:
        if row.get("source_image") is not None and row.get("target_image") is not None:
            left = self._coerce_image(row["source_image"])
            right = self._coerce_image(row["target_image"])
            return self._make_pair_canvas(left, right, "Source", "Target")
        if row.get("frames"):
            frames = row["frames"]
            return self._coerce_image(frames[len(frames) // 2])
        return self.dataset.get_image_from_row(row)

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del image
        if row is None:
            raise ValueError("MultipleChoiceBenchmark requires a dataset row.")
        choices = labels or list(self._get_choices(row))
        question = self._get_question(row)
        rendered_choices = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
        return (
            "USER: <image>\n"
            f"{self.default_instruction}\n"
            f"Question: {question}\n"
            "Choices:\n"
            f"{rendered_choices}\n"
            "Return either the choice letter or the exact choice text.\n"
            "ASSISTANT:"
        )

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        answer = self._get_answer(row)
        return [answer] if answer else []

    def evaluate_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        image: Any | None = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        del image
        choices = list(self._get_choices(row))
        answer = self._get_answer(row)
        selected = self._parse_choice(prediction=prediction, choices=choices)
        is_correct = self.dataset.normalize_text(selected) == self.dataset.normalize_text(answer)
        return (
            is_correct,
            [answer] if answer else [],
            {
                "choices": choices,
                "selected_choice": selected,
                "answer": answer,
            },
        )

    def analyze_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        prompt_labels: List[str],
        valid_labels: List[str],
        predicted_boxes: List[Dict[str, Any]],
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        del prompt_labels
        del valid_labels
        del predicted_boxes
        del row
        del evaluation
        return {
            "generated_output_count": 0 if not str(prediction or "").strip() else 1,
            "hallucinated_label_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "predicted_detection_count": 0,
        }

    def _get_question(self, row: Dict[str, Any]) -> str:
        getter = getattr(self.dataset, "get_question_from_row", None)
        if callable(getter):
            return str(getter(row))
        for key in ("question", "prompt", "instruction", "task_prompt"):
            value = row.get(key)
            if value:
                return str(value)
        return self.default_instruction

    def _get_choices(self, row: Dict[str, Any]) -> Sequence[str]:
        getter = getattr(self.dataset, "get_choices_from_row", None)
        if callable(getter):
            dataset_choices = list(getter(row))
            if dataset_choices:
                return dataset_choices
        row_choices = [str(item) for item in row.get("choices", [])]
        if row_choices:
            return row_choices
        return self._build_choices(row, [])

    def _get_answer(self, row: Dict[str, Any]) -> str:
        getter = getattr(self.dataset, "get_answer_from_row", None)
        if callable(getter):
            return str(getter(row))
        return str(row.get("answer", "")).strip()

    def _ensure_choices(self, row: Dict[str, Any], answer_pool_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if row.get("choices"):
            return row
        row["choices"] = self._build_choices(row, answer_pool_rows)
        return row

    def _build_choices(self, row: Dict[str, Any], answer_pool_rows: Sequence[Dict[str, Any]]) -> List[str]:
        answer = self._get_answer(row)
        if not answer:
            return []
        if self.fixed_choices is not None:
            choices = [str(choice) for choice in self.fixed_choices]
            if not any(self.dataset.normalize_text(choice) == self.dataset.normalize_text(answer) for choice in choices):
                choices.insert(0, answer)
            return choices

        answer_norm = self.dataset.normalize_text(answer)
        choices = [answer]
        seen = {answer_norm}
        for pool_row in answer_pool_rows:
            candidate = self._get_answer(pool_row)
            candidate_norm = self.dataset.normalize_text(candidate)
            if not candidate or candidate_norm in seen:
                continue
            seen.add(candidate_norm)
            choices.append(candidate)
            if len(choices) >= 4:
                return choices
        for candidate in self.fallback_distractors:
            candidate_norm = self.dataset.normalize_text(candidate)
            if candidate_norm in seen:
                continue
            seen.add(candidate_norm)
            choices.append(str(candidate))
            if len(choices) >= 4:
                break
        return choices

    def _parse_choice(self, prediction: str, choices: Sequence[str]) -> str:
        pred = str(prediction or "").strip()
        if not pred:
            return ""
        normalized_pred = self.dataset.normalize_text(pred)
        for choice in choices:
            if normalized_pred == self.dataset.normalize_text(choice):
                return str(choice)
        match = re.match(r"^\s*([A-Z])\b", pred)
        if match:
            index = ord(match.group(1).upper()) - 65
            if 0 <= index < len(choices):
                return str(choices[index])
        match = re.search(
            r"\b(?:answer|choice|option)\s*(?:is|:)?\s*[*_`\"']*\(?([A-Z])\)?[*_`\"']*\b",
            pred,
            re.IGNORECASE,
        )
        if match:
            index = ord(match.group(1).upper()) - 65
            if 0 <= index < len(choices):
                return str(choices[index])
        matched_choices = [
            str(choice)
            for choice in choices
            if self.dataset.normalize_text(choice) and self.dataset.normalize_text(choice) in normalized_pred
        ]
        if len(matched_choices) == 1:
            return matched_choices[0]
        return pred

    def _make_pair_canvas(self, left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
        left_rgb = left.convert("RGB")
        right_rgb = right.convert("RGB")
        width = left_rgb.width + right_rgb.width
        height = max(left_rgb.height, right_rgb.height) + 24
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(left_rgb, (0, 24))
        canvas.paste(right_rgb, (left_rgb.width, 24))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), left_label, fill=(0, 0, 0))
        draw.text((left_rgb.width + 8, 4), right_label, fill=(0, 0, 0))
        return canvas
