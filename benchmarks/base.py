from __future__ import annotations

import re
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

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

    def get_ground_truth_boxes_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        boxes: List[Dict[str, Any]] = []
        self._collect_boxes(value=row.get("objects"), out=boxes)
        self._collect_boxes(value=row.get("annotations"), out=boxes)
        self._collect_boxes(value=row.get("regions"), out=boxes)
        self._collect_boxes(value=row.get("boxes"), out=boxes)
        self._collect_boxes(value=row.get("bboxes"), out=boxes)
        self._collect_boxes(value=row.get("bbox"), out=boxes)
        return boxes

    def get_predicted_boxes(self, prediction: str) -> List[Dict[str, Any]]:
        """
        Parses outputs containing coordinate groups like:
        - [x, y, w, h]
        - (x1, y1, x2, y2)
        Optional labels are supported using:
        - "dog: [12, 10, 100, 90]"
        """
        box_pattern = re.compile(
            r"(?:(?P<label>[A-Za-z0-9 _-]{1,64})\s*:\s*)?"
            r"[\[\(]\s*"
            r"(?P<a>-?\d+(?:\.\d+)?)\s*,\s*"
            r"(?P<b>-?\d+(?:\.\d+)?)\s*,\s*"
            r"(?P<c>-?\d+(?:\.\d+)?)\s*,\s*"
            r"(?P<d>-?\d+(?:\.\d+)?)\s*"
            r"[\]\)]"
        )
        out: List[Dict[str, Any]] = []
        for m in box_pattern.finditer(prediction or ""):
            x0 = float(m.group("a"))
            y0 = float(m.group("b"))
            x1_or_w = float(m.group("c"))
            y1_or_h = float(m.group("d"))

            # Heuristic: x2/y2 usually exceed x1/y1, while w/h are positive extents.
            if x1_or_w >= x0 and y1_or_h >= y0:
                x1, y1 = x1_or_w, y1_or_h
            else:
                x1, y1 = x0 + x1_or_w, y0 + y1_or_h

            label = (m.group("label") or "").strip()
            out.append({"xyxy": [x0, y0, x1, y1], "label": label})
        return out

    def _collect_boxes(self, value: Any, out: List[Dict[str, Any]]) -> None:
        if value is None:
            return
        if self._is_numeric_box(value):
            xyxy = self._to_xyxy(value)
            if xyxy:
                out.append({"xyxy": xyxy, "label": ""})
            return
        if isinstance(value, list):
            for item in value:
                self._collect_boxes(item, out)
            return
        if isinstance(value, dict):
            # Typical dictionary form:
            # {"bbox": [x, y, w, h], "label": "cat"} or {"x":..., "y":..., "width":..., "height":...}
            label = ""
            for lk in ("label", "name", "category", "class", "entity", "phrase"):
                if lk in value and value[lk] is not None:
                    label = str(value[lk]).strip()
                    break

            if "bbox" in value:
                self._collect_box_field(box_field=value["bbox"], value=value, fallback_label=label, out=out)
            elif "box" in value:
                xyxy = self._to_xyxy(value["box"])
                if xyxy:
                    out.append({"xyxy": xyxy, "label": label})
            elif "bboxes" in value:
                self._collect_box_field(box_field=value["bboxes"], value=value, fallback_label=label, out=out)
            elif all(k in value for k in ("x", "y", "width", "height")):
                xyxy = self._to_xyxy([value["x"], value["y"], value["width"], value["height"]])
                if xyxy:
                    out.append({"xyxy": xyxy, "label": label})
            elif all(k in value for k in ("x1", "y1", "x2", "y2")):
                xyxy = self._to_xyxy([value["x1"], value["y1"], value["x2"], value["y2"]], assume_xyxy=True)
                if xyxy:
                    out.append({"xyxy": xyxy, "label": label})

            for nested in ("annotations", "objects", "regions", "boxes", "bbox", "bboxes", "entities"):
                if nested in value and value[nested] is not value:
                    self._collect_boxes(value[nested], out)
            return

    def _collect_box_field(
        self,
        box_field: Any,
        value: Dict[str, Any],
        fallback_label: str,
        out: List[Dict[str, Any]],
    ) -> None:
        if self._is_numeric_box(box_field):
            xyxy = self._to_xyxy(box_field)
            if xyxy:
                out.append({"xyxy": xyxy, "label": fallback_label})
            return

        labels_list = self._extract_label_list(value)
        if isinstance(box_field, list):
            for idx, box in enumerate(box_field):
                xyxy = self._to_xyxy(box)
                if not xyxy:
                    continue
                label = fallback_label
                if idx < len(labels_list):
                    label = labels_list[idx]
                out.append({"xyxy": xyxy, "label": label})
            return

        self._collect_boxes(box_field, out)

    def _extract_label_list(self, value: Dict[str, Any]) -> List[str]:
        for key in ("label", "labels", "name", "names", "category", "categories", "class", "classes"):
            raw = value.get(key)
            if isinstance(raw, list):
                labels: List[str] = []
                for x in raw:
                    parsed = self._stringify_label_value(x)
                    if parsed:
                        labels.append(parsed)
                return labels
        return []

    def _stringify_label_value(self, value: Any) -> str:
        if isinstance(value, int):
            labels = getattr(self.dataset, "labels", [])
            if 0 <= value < len(labels):
                return str(labels[value]).strip()
        return str(value).strip()

    def _is_numeric_box(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return False
        for v in value:
            if isinstance(v, (list, tuple, dict)):
                return False
            if not isinstance(v, (int, float)):
                return False
        return True

    def _to_xyxy(self, bbox: Any, assume_xyxy: bool = False) -> List[float] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            a, b, c, d = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None

        if assume_xyxy:
            x0, y0, x1, y1 = a, b, c, d
        else:
            # Default assumption for benchmark datasets: xywh.
            x0, y0, x1, y1 = a, b, a + c, b + d

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return [x0, y0, x1, y1]

    def run(
        self,
        model: BaseModel,
        n: int = 2,
        label_sample_size: int = 64,
        show_progress: bool = True,
        on_sample: Callable[[Dict[str, Any]], None] | None = None,
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
            gt_boxes = self.get_ground_truth_boxes_for_row(row)
            predicted_boxes = self.get_predicted_boxes(prediction)

            payload = {
                "index": idx,
                "total": total,
                "image": image,
                "prediction": prediction,
                "correct": is_correct,
                "valid_labels": valid_labels,
                "ground_truth_boxes": gt_boxes,
                "predicted_boxes": predicted_boxes,
            }

            if on_sample is not None:
                on_sample(payload)

            results.append(
                {
                    "index": idx,
                    "prediction": prediction,
                    "correct": is_correct,
                    "valid_labels": valid_labels,
                    "ground_truth_boxes": gt_boxes,
                    "predicted_boxes": predicted_boxes,
                }
            )

        return {
            "benchmark": self.name,
            "dataset": self.dataset.name,
            "num_samples": len(rows),
            "num_candidate_labels": len(labels),
            "results": results,
        }
