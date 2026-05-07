from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .._base_benchmark import BaseBenchmark


class DetectionBenchmark(BaseBenchmark):
    MAX_PROMPT_LABELS = 10
    PLACEHOLDER_LABELS = {"label", "<label_name>", "class", "object"}

    def get_predicted_boxes(self, prediction: str) -> List[Dict[str, Any]]:
        box_pattern = re.compile(
            r"(?:(?P<label>[A-Za-z0-9 _-]{1,64})\s*:\s*)?"
            r"[\[\(]\s*"
            r"(?P<x>-?\d+(?:\.\d+)?)\s*,\s*"
            r"(?P<y>-?\d+(?:\.\d+)?)\s*,\s*"
            r"(?P<w>-?\d+(?:\.\d+)?)\s*,\s*"
            r"(?P<h>-?\d+(?:\.\d+)?)\s*"
            r"[\]\)]"
        )
        out: List[Dict[str, Any]] = []
        for match in box_pattern.finditer(prediction or ""):
            x = float(match.group("x"))
            y = float(match.group("y"))
            w = float(match.group("w"))
            h = float(match.group("h"))
            out.append(
                {
                    "label": (match.group("label") or "").strip(),
                    "xyxy": [x, y, x + w, y + h],
                }
            )
        return out

    def _coerce_label(self, value: Any) -> str | None:
        if isinstance(value, int):
            category_map = getattr(self.dataset, "category_id_to_label", {})
            if value in category_map:
                return category_map[value]
            if 0 <= value < len(self.dataset.labels):
                return self.dataset.labels[value]
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("label", "name", "category", "category_name", "class", "category_id"):
                if key in value:
                    return self._coerce_label(value[key])
        return None

    def _flatten_labels(self, value: Any) -> List[str]:
        out: List[str] = []
        if isinstance(value, list):
            for item in value:
                out.extend(self._flatten_labels(item))
            return out
        coerced = self._coerce_label(value)
        return [coerced] if coerced else []

    def _labels_from_row_annotations(self, row: Dict[str, Any]) -> List[str]:
        labels: Set[str] = set()
        containers = []
        for key in ("objects", "annotations", "instances", "segments_info"):
            if key in row and row[key] is not None:
                containers.append(row[key])

        for container in containers:
            if isinstance(container, dict):
                for key in (
                    "category",
                    "categories",
                    "category_id",
                    "category_ids",
                    "label",
                    "labels",
                    "name",
                    "names",
                    "class",
                    "classes",
                ):
                    if key in container:
                        labels.update(self._flatten_labels(container[key]))
            else:
                labels.update(self._flatten_labels(container))

        return sorted(labels)

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        ann_labels = self.dataset.get_labels_img(row)
        if ann_labels:
            return ann_labels
        return self._labels_from_row_annotations(row)

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        labels: Set[str] = set()
        for row in rows:
            labels.update(self._labels_from_row_annotations(row))
        if labels:
            return sorted(labels)
        return list(getattr(self.dataset, "labels", []))

    def get_ground_truth_boxes_for_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        getter = getattr(self.dataset, "get_annotations_for_row", None)
        if callable(getter):
            annotations = getter(row)
            if annotations:
                out: List[Dict[str, Any]] = []
                for item in annotations:
                    xyxy = self._to_xyxy(item.get("bbox"), assume_xyxy=False)
                    if xyxy is None:
                        continue
                    out.append({"label": str(item.get("label", "")).strip(), "xyxy": xyxy})
                return out

        by_coords: Dict[tuple[float, float, float, float], Dict[str, Any]] = {}
        for box in super().get_ground_truth_boxes_for_row(row):
            xyxy = box.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            coords = tuple(float(value) for value in xyxy)
            label = str(box.get("label", "")).strip()
            existing = by_coords.get(coords)
            if existing is None or (label and not existing["label"]):
                by_coords[coords] = {"label": label, "xyxy": list(coords)}
        return list(by_coords.values())

    def postprocess_ground_truth_boxes(
        self,
        gt_boxes: List[Dict[str, Any]],
        image: Any | None = None,
    ) -> List[Dict[str, Any]]:
        if image is None or not hasattr(image, "size"):
            return gt_boxes

        width, height = image.size
        out: List[Dict[str, Any]] = []
        for box in gt_boxes:
            xyxy = box.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in xyxy]
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
                x0 *= width
                x1 *= width
                y0 *= height
                y1 *= height
            out.append({"label": box.get("label", ""), "xyxy": [x0, y0, x1, y1]})
        return out

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        valid = self.get_valid_labels_for_row(row)
        if valid:
            return valid[: self.MAX_PROMPT_LABELS]
        return labels[: self.MAX_PROMPT_LABELS]

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del row
        del image
        label_text = ", ".join(labels)
        example_label = labels[0] if labels else "object"
        return (
            "USER: <image>\n"
            "Detect the objects in the image.\n"
            "Return one line per detection using this exact format:\n"
            f"{example_label}: [0.1, 0.2, 0.3, 0.4]\n"
            "Coordinates must be normalized to 0..1.\n"
            f"Only use labels from this list: {label_text}\n"
            "Do not describe the image. Do not write full sentences.\n"
            "ASSISTANT:"
        )

    def postprocess_predicted_boxes(
        self,
        pred_boxes: List[Dict[str, Any]],
        image: Any | None = None,
    ) -> List[Dict[str, Any]]:
        if image is None or not hasattr(image, "size"):
            return pred_boxes

        width, height = image.size
        out: List[Dict[str, Any]] = []
        for box in pred_boxes:
            xyxy = box.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in xyxy]
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
                x0 *= width
                x1 *= width
                y0 *= height
                y1 *= height
            out.append({"label": box.get("label", ""), "xyxy": [x0, y0, x1, y1]})
        return out

    def _filter_prediction_boxes(
        self,
        row: Dict[str, Any],
        labels: List[str],
        predicted_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del row
        allowed = {self.dataset.normalize_text(label): label for label in labels}
        out: List[Dict[str, Any]] = []
        for box in predicted_boxes:
            xyxy = box.get("xyxy")
            if xyxy is None:
                continue
            label = str(box.get("label", "")).strip()
            if self.dataset.normalize_text(label) in self.PLACEHOLDER_LABELS:
                label = ""
            if not label and len(labels) == 1:
                label = str(labels[0]).strip()
            norm = self.dataset.normalize_text(label)
            if norm not in allowed:
                continue
            out.append({"label": allowed[norm], "xyxy": xyxy})
        return out

    def evaluate_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        image: Any | None = None,
    ):
        valid_labels = self.get_valid_labels_for_row(row)
        prompt_labels = self.get_prompt_labels_for_row(row, valid_labels)
        gt_boxes = self.get_ground_truth_boxes_for_row(row)
        gt_boxes = self.postprocess_ground_truth_boxes(gt_boxes, image=image)
        pred_boxes = self.parse_prediction_boxes(prediction, image=image)
        pred_boxes = self._filter_prediction_boxes(row, prompt_labels, pred_boxes)

        matched_predictions, matched_targets, matched_ious = self._match_boxes(pred_boxes, gt_boxes)
        tp = len(matched_predictions)
        fp = max(0, len(pred_boxes) - tp)
        fn = max(0, len(gt_boxes) - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        mean_iou_matched = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        mean_iou_all_predictions = sum(matched_ious) / len(pred_boxes) if pred_boxes else 0.0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_iou_matched": mean_iou_matched,
            "mean_iou_all_predictions": mean_iou_all_predictions,
            "ground_truth_boxes": gt_boxes,
            "predicted_boxes": pred_boxes,
            "matched_predictions": matched_predictions,
            "matched_targets": matched_targets,
        }
        correct = tp > 0 and fn == 0 and fp == 0
        return correct, valid_labels, metrics

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
        del valid_labels
        raw_boxes = self.parse_prediction_boxes(prediction=prediction)
        allowed = {self.dataset.normalize_text(label) for label in prompt_labels}
        hallucinated_label_count = 0
        for box in raw_boxes:
            label = self.dataset.normalize_text(box.get("label", ""))
            if label and label not in allowed and label not in self.PLACEHOLDER_LABELS:
                hallucinated_label_count += 1
        return {
            "generated_output_count": len(raw_boxes),
            "hallucinated_label_count": hallucinated_label_count,
            "false_positive_count": int(evaluation.get("fp", 0)),
            "false_negative_count": int(evaluation.get("fn", 0)),
            "predicted_detection_count": len(predicted_boxes),
        }

    def _match_boxes(self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]], iou_threshold: float = 0.5):
        matched_predictions: List[Dict[str, Any]] = []
        matched_targets: List[Dict[str, Any]] = []
        matched_ious: List[float] = []
        used_targets: Set[int] = set()

        for pred in predictions:
            pred_label = self.dataset.normalize_text(pred.get("label", ""))
            pred_xyxy = pred.get("xyxy")
            if pred_xyxy is None:
                continue

            best_idx = None
            best_iou = 0.0
            for idx, target in enumerate(targets):
                if idx in used_targets:
                    continue
                target_label = self.dataset.normalize_text(target.get("label", ""))
                if pred_label != target_label:
                    continue
                target_xyxy = target.get("xyxy")
                if target_xyxy is None:
                    continue
                iou = self._iou(pred_xyxy, target_xyxy)
                if iou >= iou_threshold and iou > best_iou:
                    best_idx = idx
                    best_iou = iou

            if best_idx is None:
                continue

            used_targets.add(best_idx)
            matched_predictions.append(pred)
            matched_targets.append(targets[best_idx])
            matched_ious.append(best_iou)

        return matched_predictions, matched_targets, matched_ious

    def _iou(self, a: List[float] | tuple[float, float, float, float], b: List[float] | tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union
