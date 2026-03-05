from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Tuple

from models.base import BaseModel


class BaseBenchmark(ABC):
    name: str

    def __init__(self, dataset, name: str):
        self.dataset = dataset
        self.name = name

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        return self.dataset.get_labels(rows)

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        return self.dataset.get_labels_img(row)

    def make_prompt(self, labels: List[str]) -> str:
        label_set = ", ".join(labels)
        return (
            "USER: <image>\n"
            "Return exactly ONE label from this list (one item only, no extra words):\n"
            f"{label_set}\n"
            "ASSISTANT:"
        )

    def prepare(self, n: int, label_sample_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        rows_for_labels = self.dataset.get_samples(max(n, label_sample_size))
        rows = rows_for_labels[:n]
        labels = self.get_candidate_labels(rows_for_labels)
        return rows, labels

    def evaluate_prediction(self, row: Dict[str, Any], prediction: str) -> Tuple[bool, List[str]]:
        valid_labels = self.get_valid_labels_for_row(row)
        valid_labels_norm = sorted({self.dataset.normalize_text(l) for l in valid_labels})
        pred_norm = self.dataset.normalize_text(prediction)
        return pred_norm in set(valid_labels_norm), valid_labels_norm

    def run(
        self,
        model: BaseModel,
        n: int = 2,
        label_sample_size: int = 64,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        rows, labels = self.prepare(n=n, label_sample_size=label_sample_size)
        prompt = self.make_prompt(labels)

        results: List[Dict[str, Any]] = []
        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            if show_progress and total > 0:
                pct = (idx / total) * 100.0
                print(f"Progress: {pct:.1f}% ({idx}/{total})")

            image = self.dataset.get_image_from_row(row)
            prediction = model.predict(image, prompt).strip()
            is_correct, valid_labels = self.evaluate_prediction(row, prediction)
            results.append(
                {
                    "index": idx,
                    "prediction": prediction,
                    "correct": is_correct,
                    "valid_labels": valid_labels,
                }
            )

        return {
            "benchmark": self.name,
            "dataset": self.dataset.name,
            "num_samples": len(rows),
            "num_candidate_labels": len(labels),
            "results": results,
        }
