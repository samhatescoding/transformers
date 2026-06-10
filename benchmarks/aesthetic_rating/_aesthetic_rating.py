from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .._base_benchmark import BaseBenchmark


class AestheticRatingBenchmark(BaseBenchmark):
    """Predict an image's rounded aesthetic rating on a 1-10 scale."""

    task_type = "aesthetic_rating"
    default_max_new_tokens = 4

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        del rows
        return [str(value) for value in range(1, 11)]

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        del row
        return labels

    def make_prompt(self, labels, row=None, image=None) -> str:
        del labels
        del row
        del image
        return (
            "USER: <image>\n"
            "Rate the overall aesthetic quality of this image from 1 to 10, "
            "where 1 is very poor and 10 is exceptional.\n"
            "Return exactly one integer from 1 to 10.\n"
            "ASSISTANT:"
        )

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        rating = self._get_rating(row)
        return [str(rating)] if rating is not None else []

    def evaluate_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        image: Any | None = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        del image
        target = self._get_rating(row)
        predicted = self._parse_rating(prediction)
        error = abs(predicted - target) if predicted is not None and target is not None else None
        return (
            predicted == target and target is not None,
            [str(target)] if target is not None else [],
            {
                "predicted_rating": predicted,
                "target_rating": target,
                "absolute_error": error,
            },
        )

    def build_run_statistics(self, sample_results, wall_clock_seconds):
        stats = super().build_run_statistics(sample_results, wall_clock_seconds)
        errors = [item.get("absolute_error") for item in sample_results if item.get("absolute_error") is not None]
        stats["mean_absolute_error"] = float(sum(errors) / len(errors)) if errors else None
        return stats

    def _get_rating(self, row: Dict[str, Any]) -> int | None:
        getter = getattr(self.dataset, "get_rating_from_row", None)
        value = getter(row) if callable(getter) else row.get("rating", row.get("score"))
        if value is None:
            return None
        return max(1, min(10, int(float(value) + 0.5)))

    @staticmethod
    def _parse_rating(prediction: str) -> int | None:
        match = re.search(r"(?<!\d)(10|[1-9])(?!\d)", str(prediction or ""))
        return int(match.group(1)) if match else None
