from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw


class BenchmarkSampleSaver:
    """
    Saves benchmark samples to disk for post-run inspection.
    """

    def __init__(self, base_dir: str = "ui_outputs", run_name: str | None = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = run_name or f"run_{timestamp}"
        self.output_dir = Path(base_dir) / folder_name
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
        self._draw_boxes(draw, gt_boxes, color="lime")
        self._draw_boxes(draw, pred_boxes, color="orange")

        image_name = f"sample_{idx:04d}.png"
        image_path = self.output_dir / image_name
        annotated.save(image_path)

        record = {
            "index": idx,
            "total": payload.get("total"),
            "prediction": payload.get("prediction", ""),
            "correct": payload.get("correct", False),
            "valid_labels": payload.get("valid_labels", []),
            "ground_truth_boxes": gt_boxes,
            "predicted_boxes": pred_boxes,
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

    def _draw_boxes(self, draw: ImageDraw.ImageDraw, boxes: List[Dict[str, Any]], color: str) -> None:
        for item in boxes or []:
            if not isinstance(item, dict):
                continue
            coords = item.get("xyxy")
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in coords]
            except (TypeError, ValueError):
                continue
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            label = str(item.get("label", "")).strip()
            if label:
                text_y = max(0, y0 - 15)
                draw.text((x0 + 2, text_y), label, fill=color)
