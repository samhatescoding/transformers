from __future__ import annotations

import random
import re
import time
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

from models._base_model import BaseModel
from benchmarks.resource_sampler import ResourceSampler
from dataset._base_dataset import BaseDataset


class BaseBenchmark(ABC):
    name: str
    dataset: BaseDataset
    MAX_PROMPT_LABELS = 16

    def __init__(self, dataset: BaseDataset, name: str):
        self.dataset = dataset
        self.name = name

    def get_candidate_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        return self.dataset.get_labels(rows)

    def get_valid_labels_for_row(self, row: Dict[str, Any]) -> List[str]:
        return self.dataset.get_labels_img(row)

    def get_prompt_labels_for_row(self, row: Dict[str, Any], labels: List[str]) -> List[str]:
        if len(labels) <= self.MAX_PROMPT_LABELS:
            return labels

        valid_labels = self.get_valid_labels_for_row(row)
        valid_norms = {self.dataset.normalize_text(label) for label in valid_labels}
        chosen: List[str] = []
        seen = set()
        for label in labels:
            norm = self.dataset.normalize_text(label)
            if norm in valid_norms and norm not in seen:
                seen.add(norm)
                chosen.append(label)

        remaining = [label for label in labels if self.dataset.normalize_text(label) not in seen]
        rng = self.make_rng_for_row(row)
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, self.MAX_PROMPT_LABELS - len(chosen))])
        return chosen[: self.MAX_PROMPT_LABELS]

    def make_prompt(
        self,
        labels: List[str],
        row: Dict[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        del row
        del image
        label_set = ", ".join(labels)
        return (
            "USER: <image>\n"
            "Return exactly ONE label from this list (one item only, no extra words):\n"
            f"{label_set}\n"
            "ASSISTANT:"
        )

    def get_image_for_row(self, row: Dict[str, Any]) -> Any:
        return self.dataset.get_image_from_row(row)

    def make_rng_for_row(self, row: Dict[str, Any]) -> random.Random:
        seed_parts = [
            self.name,
            getattr(self.dataset, "name", ""),
            row.get("id"),
            row.get("image_id"),
            row.get("clip_id"),
            row.get("video_id"),
            row.get("file_name"),
        ]
        return random.Random("|".join("" if part is None else str(part) for part in seed_parts))

    def prepare(self, n: int, label_sample_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        rows_for_labels = self.dataset.get_samples(max(n, label_sample_size))
        rows = rows_for_labels[:n]
        labels = self.get_candidate_labels(rows_for_labels)
        return rows, labels

    def evaluate_prediction(
        self,
        row: Dict[str, Any],
        prediction: str,
        image: Any | None = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        del image
        valid_labels = self.get_valid_labels_for_row(row)
        valid_labels_norm = sorted({self.dataset.normalize_text(l) for l in valid_labels})
        pred_norm = self.dataset.normalize_text(prediction)
        return pred_norm in set(valid_labels_norm), valid_labels_norm, {}

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

    def postprocess_predicted_boxes(
        self,
        pred_boxes: List[Dict[str, Any]],
        image: Any | None = None,
    ) -> List[Dict[str, Any]]:
        del image
        return pred_boxes

    def parse_prediction_boxes(
        self,
        prediction: str,
        image: Any | None = None,
    ) -> List[Dict[str, Any]]:
        pred_boxes = self.get_predicted_boxes(prediction)
        return self.postprocess_predicted_boxes(pred_boxes=pred_boxes, image=image)

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
            for lk in ("label", "name", "category", "class", "entity", "phrase", "category_id"):
                if lk in value and value[lk] is not None:
                    label = self._stringify_label_value(value[lk])
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
        for key in (
            "label",
            "labels",
            "name",
            "names",
            "category",
            "categories",
            "class",
            "classes",
            "category_id",
            "category_ids",
        ):
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
            category_map = getattr(self.dataset, "category_id_to_label", {})
            if value in category_map:
                return str(category_map[value]).strip()
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
        normalized_prompt_labels = {self.dataset.normalize_text(label) for label in prompt_labels}
        normalized_prediction = self.dataset.normalize_text(prediction)
        hallucinated_label_count = 1 if bool(normalized_prediction) and normalized_prediction not in normalized_prompt_labels else 0
        return {
            "generated_output_count": 0 if not str(prediction or "").strip() else 1,
            "hallucinated_label_count": hallucinated_label_count,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "predicted_detection_count": len(predicted_boxes),
        }

    def build_run_statistics(
        self,
        sample_results: List[Dict[str, Any]],
        wall_clock_seconds: float,
    ) -> Dict[str, Any]:
        successful = [item for item in sample_results if item["stats"]["success"]]
        completed = list(sample_results)
        successful_count = len(successful)
        failure_count = len(sample_results) - successful_count

        def mean_of(key: str) -> float | None:
            values = [item["stats"].get(key) for item in completed if item["stats"].get(key) is not None]
            if not values:
                return None
            return float(sum(values) / len(values))

        def peak_of(key: str) -> int | None:
            values = [item["stats"].get(key) for item in completed if item["stats"].get(key) is not None]
            if not values:
                return None
            return int(max(values))

        truncated = sum(1 for item in completed if item["stats"]["truncated"])
        hallucinated_samples = sum(1 for item in completed if item["stats"]["hallucinated_label_count"] > 0)
        false_positives = sum(int(item["stats"]["false_positive_count"]) for item in completed)
        false_negatives = sum(int(item["stats"]["false_negative_count"]) for item in completed)
        predicted_detections = [int(item["stats"]["predicted_detection_count"]) for item in completed]
        retry_count = sum(int(item["stats"]["retry_count"]) for item in sample_results)
        output_tokens = [item["stats"]["output_tokens"] for item in completed if item["stats"]["output_tokens"] is not None]
        generation_times = [
            item["stats"]["generation_time_seconds"]
            for item in completed
            if item["stats"]["generation_time_seconds"] is not None
        ]
        completed_count = len(completed)

        return {
            "wall_clock_time_seconds": wall_clock_seconds,
            "wall_clock_time_per_sample_seconds_mean": mean_of("wall_clock_time_seconds"),
            "total_generation_time_seconds_mean": mean_of("generation_time_seconds"),
            "first_token_latency_seconds_mean": mean_of("first_token_latency_seconds"),
            "time_per_output_token_seconds_mean": mean_of("time_per_output_token_seconds"),
            "samples_per_second": (completed_count / wall_clock_seconds) if wall_clock_seconds > 0 else None,
            "number_of_output_tokens_mean": mean_of("output_tokens"),
            "number_of_generated_outputs_mean": mean_of("generated_output_count"),
            "number_of_benchmark_samples_completed": successful_count,
            "success_count": successful_count,
            "failure_count": failure_count,
            "retry_count": retry_count,
            "peak_cpu_ram_bytes": peak_of("peak_cpu_ram_bytes"),
            "peak_gpu_memory_bytes": peak_of("peak_gpu_memory_bytes"),
            "cpu_utilization_percent_mean": mean_of("cpu_utilization_percent"),
            "gpu_utilization_percent_mean": None,
            "vram_allocation_over_time_bytes": None,
            "disk_offload_volume_bytes": None,
            "truncation_rate": (truncated / completed_count) if completed_count else None,
            "hallucinated_label_rate": (hallucinated_samples / completed_count) if completed_count else None,
            "false_positive_count_total": false_positives,
            "false_negative_count_total": false_negatives,
            "mean_number_of_predicted_detections": (
                sum(predicted_detections) / len(predicted_detections) if predicted_detections else None
            ),
            "tokens_per_second": (
                (sum(output_tokens) / sum(generation_times))
                if output_tokens and generation_times and sum(generation_times) > 0
                else None
            ),
        }

    def run(
        self,
        model: BaseModel,
        n: int = 2,
        label_sample_size: int = 64,
        show_progress: bool = True,
        on_sample: Callable[[Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        rows, labels = self.prepare(n=n, label_sample_size=label_sample_size)

        results: List[Dict[str, Any]] = []
        total = len(rows)
        run_started_at = time.perf_counter()
        for idx, row in enumerate(rows, start=1):
            if show_progress and total > 0:
                pct = (idx / total) * 100.0
                print(f"Progress: {pct:.1f}% ({idx}/{total})")

            sample_started_at = time.perf_counter()
            image = None
            prompt_labels: List[str] = []
            prediction = ""
            sample_error: str | None = None
            generation_time_seconds = 0.0
            resource_stats = {
                "peak_cpu_ram_bytes": None,
                "cpu_utilization_percent": None,
                "peak_gpu_memory_bytes": None,
                "gpu_utilization_percent": None,
                "vram_allocation_over_time_bytes": None,
                "disk_offload_volume_bytes": None,
            }

            try:
                image = self.get_image_for_row(row)
                prompt_labels = self.get_prompt_labels_for_row(row=row, labels=labels)
                prompt = self.make_prompt(labels=prompt_labels, row=row, image=image)
                sampler = ResourceSampler()
                sampler.start()

                generation_started_at = time.perf_counter()
                try:
                    prediction = model.predict(image, prompt).strip()
                except Exception as exc:
                    sample_error = f"{exc.__class__.__name__}: {exc}"
                generation_finished_at = time.perf_counter()
                resource_stats = sampler.stop()
                generation_time_seconds = generation_finished_at - generation_started_at
            except Exception as exc:
                sample_error = f"{exc.__class__.__name__}: {exc}"

            wall_clock_time_seconds = time.perf_counter() - sample_started_at

            predicted_boxes = self.parse_prediction_boxes(prediction=prediction, image=image)
            if sample_error is None:
                is_correct, valid_labels, evaluation = self.evaluate_prediction(row, prediction, image=image)
            else:
                is_correct = False
                valid_labels = self.get_valid_labels_for_row(row)
                evaluation = {"error": sample_error}
            gt_boxes = self.get_ground_truth_boxes_for_row(row)
            behavior_stats = self.analyze_prediction(
                row=row,
                prediction=prediction,
                prompt_labels=prompt_labels,
                valid_labels=valid_labels,
                predicted_boxes=predicted_boxes,
                evaluation=evaluation,
            )

            token_counter = getattr(model, "count_text_tokens", None)
            output_tokens = token_counter(prediction) if callable(token_counter) else None
            time_per_output_token_seconds = None
            if output_tokens and generation_time_seconds > 0:
                time_per_output_token_seconds = generation_time_seconds / output_tokens

            sample_stats = {
                "success": sample_error is None,
                "error": sample_error,
                "wall_clock_time_seconds": wall_clock_time_seconds,
                "generation_time_seconds": generation_time_seconds,
                "first_token_latency_seconds": None,
                "time_per_output_token_seconds": time_per_output_token_seconds,
                "output_tokens": output_tokens,
                "retry_count": 0,
                "truncated": bool(
                    output_tokens is not None
                    and getattr(model, "max_new_tokens", None) is not None
                    and output_tokens >= int(getattr(model, "max_new_tokens"))
                ),
            }
            sample_stats.update(resource_stats)
            sample_stats.update(behavior_stats)

            payload = {
                "index": idx,
                "total": total,
                "image": image,
                "prompt_labels": prompt_labels,
                "prediction": prediction,
                "correct": is_correct,
                "valid_labels": valid_labels,
                "ground_truth_boxes": gt_boxes,
                "predicted_boxes": predicted_boxes,
                "stats": sample_stats,
            }
            payload.update(evaluation)

            if on_sample is not None:
                on_sample(payload)

            result_item = {
                "index": idx,
                "prompt_labels": prompt_labels,
                "prediction": prediction,
                "correct": is_correct,
                "valid_labels": valid_labels,
                "ground_truth_boxes": gt_boxes,
                "predicted_boxes": predicted_boxes,
                "stats": sample_stats,
            }
            result_item.update(evaluation)
            results.append(result_item)

        wall_clock_seconds = time.perf_counter() - run_started_at
        return {
            "benchmark": self.name,
            "dataset": self.dataset.name,
            "num_samples": len(rows),
            "num_candidate_labels": len(labels),
            "results": results,
            "stats": self.build_run_statistics(sample_results=results, wall_clock_seconds=wall_clock_seconds),
        }
