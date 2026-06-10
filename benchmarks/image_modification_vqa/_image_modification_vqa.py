from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from ..curated_distractors import get_curated_distractors
from ..multiple_choice._multiple_choice import MultipleChoiceBenchmark


class ImageModificationVQABenchmark(MultipleChoiceBenchmark):
    """Identify which instruction transformed a source image into a target image."""

    task_type = "image_modification_vqa"
    default_instruction = "Identify the instruction that transformed Image A into Image B."
    choice_count = 4

    def prepare(self, n: int, label_sample_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        rows, labels = super().prepare(n=n, label_sample_size=label_sample_size)
        return [
            self._finalize_prompt_choices(row, row_index)
            for row_index, row in enumerate(rows)
        ], labels

    def make_prompt(self, labels, row=None, image=None) -> str:
        del image
        if row is None:
            raise ValueError("ImageModificationVQABenchmark requires a dataset row.")
        choices = labels or list(self._get_choices(row))
        rendered = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
        return (
            "USER: <image>\n"
            "The left panel is Image A (before) and the right panel is Image B (after).\n"
            "Which one of the following instructions was applied to Image A to produce Image B?\n"
            f"{rendered}\n"
            "Return only the choice letter or the exact instruction.\n"
            "ASSISTANT:"
        )

    def _finalize_prompt_choices(self, row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        answer = self._get_answer(row)
        answer_norm = self.dataset.normalize_text(answer)
        distractors = []
        seen = {answer_norm}
        try:
            curated = get_curated_distractors(self.dataset.name, row_index, answer)
        except ValueError:
            curated = ()
        for choice in [*curated, *self._get_choices(row), *self.fallback_distractors]:
            normalized = self.dataset.normalize_text(choice)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            distractors.append(str(choice))
        rng = self.make_rng_for_row(row)
        choices = [answer, *distractors[: self.choice_count - 1]]
        if len(choices) != self.choice_count:
            raise ValueError(
                f"{self.name} requires exactly {self.choice_count} edit instructions per sample; got {len(choices)}."
            )
        rng.shuffle(choices)
        row["choices"] = choices
        return row
