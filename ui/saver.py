from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw


class BenchmarkSampleSaver:
    """
    Saves benchmark samples to disk for post-run inspection.
    """

    def __init__(
        self,
        base_dir: str = "ui/ui_outputs",
        run_name: str | None = None,
        clear_existing_runs: bool = False,
    ):
        self.base_dir = Path(base_dir)
        if clear_existing_runs and self.base_dir.exists():
            self._clear_base_dir()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = run_name or f"run_{timestamp}"
        self.output_dir = self.base_dir / folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._summary_path = self.output_dir / "summary.jsonl"
        self._summary_file = self._summary_path.open("a", encoding="utf-8")

    def on_sample(self, payload: Dict[str, Any]) -> None:
        idx = int(payload.get("index", 0))
        image = payload["image"].convert("RGB")
        gt_boxes = payload.get("ground_truth_boxes", [])
        pred_boxes = payload.get("predicted_boxes", [])

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        self._draw_boxes(draw, gt_boxes, default_color="lime", show_coords=False)
        self._draw_boxes(
            draw,
            pred_boxes,
            default_color="orange",
            show_coords=True,
            color_cycle=self.PRED_COLORS,
        )

        image_name = f"sample_{idx:04d}.png"
        image_path = self.output_dir / image_name
        annotated.save(image_path)

        record = {
            "index": idx,
            "total": payload.get("total"),
            "prompt_labels": payload.get("prompt_labels", []),
            "prediction": payload.get("prediction", ""),
            "correct": payload.get("correct", False),
            "valid_labels": payload.get("valid_labels", []),
            "ground_truth_boxes": gt_boxes,
            "predicted_boxes": pred_boxes,
            "box_matches": payload.get("box_matches", []),
            "matched_predictions": payload.get("matched_predictions"),
            "total_predictions": payload.get("total_predictions"),
            "total_ground_truth_boxes": payload.get("total_ground_truth_boxes"),
            "precision": payload.get("precision"),
            "recall": payload.get("recall"),
            "f1": payload.get("f1"),
            "mean_iou_matched": payload.get("mean_iou_matched"),
            "mean_iou_all_predictions": payload.get("mean_iou_all_predictions"),
            "image_file": image_name,
        }

        sample_json_path = self.output_dir / f"sample_{idx:04d}.json"
        sample_json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        self._summary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._summary_file.flush()

    def close(self) -> None:
        try:
            self._summary_file.close()
        except Exception:
            pass

    def _clear_base_dir(self) -> None:
        for item in self.base_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink()
                except OSError:
                    pass

    def _draw_boxes(
        self,
        draw: ImageDraw.ImageDraw,
        boxes: List[Dict[str, Any]],
        default_color: str,
        show_coords: bool,
        color_cycle: tuple[str, ...] | None = None,
    ) -> None:
        for idx, item in enumerate(boxes or []):
            if not isinstance(item, dict):
                continue
            coords = item.get("xyxy")
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in coords]
            except (TypeError, ValueError):
                continue
            color = color_cycle[idx % len(color_cycle)] if color_cycle else default_color
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            label = str(item.get("label", "")).strip()
            w = int(round(max(0.0, x1 - x0)))
            h = int(round(max(0.0, y1 - y0)))
            xi = int(round(x0))
            yi = int(round(y0))
            coords_text = f"[{xi}, {yi}, {w}, {h}]"
            caption = f"{label} {coords_text}".strip() if show_coords else label
            if caption:
                text_y = max(0, y0 - 15)
                draw.text((x0 + 2, text_y), caption, fill=color)
    PRED_COLORS = (
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "red",
        "dodgerblue",
        "white",
    )
