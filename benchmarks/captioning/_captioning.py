from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from .._base_benchmark import BaseBenchmark


class CaptioningBenchmark(BaseBenchmark):
    default_max_new_tokens = 24

    def __init__(self, dataset=None, name: str | None = None, bleu_threshold: float = 0.25, **dataset_kwargs):
        super().__init__(dataset=dataset, name=name, **dataset_kwargs)
        self.bleu_threshold = float(bleu_threshold)

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        del rows
        return []

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        del row
        del labels
        return []

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        return self._get_captions(row)

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del labels
        del row
        del image
        return (
            "USER: <image>\n"
            "Write one concise caption that describes the image or representative video frame.\n"
            "Return only the caption text.\n"
            "ASSISTANT:"
        )

    def evaluate_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        image: Any | None = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        del image
        references = self._get_captions(row)
        bleu = self._sentence_bleu(references=references, candidate=prediction)
        return (
            bleu >= self.bleu_threshold,
            references,
            {
                "bleu": bleu,
                "reference_captions": references,
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

    def _get_captions(self, row: Dict[str, Any]) -> List[str]:
        getter = getattr(self.dataset, "get_captions_from_row", None)
        if callable(getter):
            return list(getter(row))
        captions = row.get("captions")
        if isinstance(captions, list):
            return [str(item) for item in captions]
        caption = row.get("caption")
        if isinstance(caption, list):
            return [str(item) for item in caption]
        if caption:
            return [str(caption)]
        return []

    def _sentence_bleu(self, references: Sequence[str], candidate: str, max_n: int = 4) -> float:
        ref_tokens = [self._tokenize(text) for text in references if self._tokenize(text)]
        cand_tokens = self._tokenize(candidate)
        if not ref_tokens or not cand_tokens:
            return 0.0

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            cand_counts = self._ngram_counts(cand_tokens, n)
            if not cand_counts:
                precisions.append(1.0)
                continue
            max_ref_counts: Counter[tuple[str, ...]] = Counter()
            for ref in ref_tokens:
                ref_counts = self._ngram_counts(ref, n)
                for gram, count in ref_counts.items():
                    if count > max_ref_counts[gram]:
                        max_ref_counts[gram] = count
            clipped = sum(min(count, max_ref_counts[gram]) for gram, count in cand_counts.items())
            total = sum(cand_counts.values())
            precisions.append((clipped + 1.0) / (total + 1.0))

        cand_len = len(cand_tokens)
        ref_lens = [len(ref) for ref in ref_tokens]
        closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - cand_len), ref_len))
        brevity_penalty = 1.0 if cand_len > closest_ref_len else math.exp(1.0 - (closest_ref_len / cand_len))
        log_precision_sum = sum((1.0 / max_n) * math.log(p) for p in precisions)
        return brevity_penalty * math.exp(log_precision_sum)

    def _tokenize(self, text: str) -> List[str]:
        normalized = self.dataset.normalize_text(text)
        return normalized.split() if normalized else []

    def _ngram_counts(self, tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
