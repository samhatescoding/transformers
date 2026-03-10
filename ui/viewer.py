from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageTk


@dataclass
class Box:
    label: str
    xyxy: tuple[float, float, float, float]
    color: str


class BenchmarkUI:
    """
    Tiny Tkinter UI for visualizing benchmark samples as they are evaluated.
    """

    def __init__(self, title: str = "Benchmark Viewer", max_image_size: tuple[int, int] = (920, 620)):
        self.max_image_size = max_image_size
        self.enabled = True

        try:
            self.root = tk.Tk()
        except tk.TclError:
            # Headless environment: disable visualization gracefully.
            self.enabled = False
            self.root = None
            self._photo = None
            return

        self.root.title(title)
        self.root.geometry("1200x820")
        self.root.state("normal")
        self.root.deiconify()

        self.samples: List[Dict[str, Any]] = []
        self.current_index = -1

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

        self.prev_button = tk.Button(top, text="Previous", command=self.show_previous, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(top, text="Next", command=self.show_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=(8, 0))

        self.nav_var = tk.StringVar(value="No samples yet")
        self.nav_label = tk.Label(top, textvariable=self.nav_var, anchor="w")
        self.nav_label.pack(side=tk.LEFT, padx=12)

        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        self.image_label = tk.Label(left, text="Waiting for samples...")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.meta_var = tk.StringVar(value="")
        self.meta_label = tk.Label(
            right,
            textvariable=self.meta_var,
            justify=tk.LEFT,
            anchor="nw",
            wraplength=340,
            font=("Consolas", 10),
        )
        self.meta_label.pack(fill=tk.BOTH, expand=True)

        self._photo = None
        self.root.bind("<Left>", lambda _e: self.show_previous())
        self.root.bind("<Right>", lambda _e: self.show_next())
        self._bring_to_front()
        self.root.update_idletasks()
        self.root.update()

    def on_sample(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.root is None:
            return

        self.samples.append(payload)
        self.current_index = len(self.samples) - 1
        self._render_current_sample()
        self._bring_to_front()
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        if not self.enabled or self.root is None:
            return
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def wait_until_closed(self) -> None:
        if not self.enabled or self.root is None:
            return
        try:
            self.root.mainloop()
        except tk.TclError:
            pass

    def show_previous(self) -> None:
        if not self.samples or self.current_index <= 0:
            return
        self.current_index -= 1
        self._render_current_sample()

    def show_next(self) -> None:
        if not self.samples or self.current_index >= len(self.samples) - 1:
            return
        self.current_index += 1
        self._render_current_sample()

    def _render_image(self, image: Image.Image, gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]]) -> Image.Image:
        out = image.convert("RGB").copy()
        draw = ImageDraw.Draw(out)

        normalized_gt = self._normalize_boxes(gt_boxes, color="lime")
        normalized_pred = self._normalize_boxes(pred_boxes, color="orange")

        for box in normalized_gt + normalized_pred:
            x0, y0, x1, y1 = box.xyxy
            draw.rectangle((x0, y0, x1, y1), outline=box.color, width=3)
            if box.label:
                text_y = max(0, y0 - 15)
                draw.text((x0 + 2, text_y), box.label, fill=box.color)

        out.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
        return out

    def _normalize_boxes(self, boxes: List[Dict[str, Any]], color: str) -> List[Box]:
        out: List[Box] = []
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
            label = str(item.get("label", "")).strip()
            out.append(Box(label=label, xyxy=(x0, y0, x1, y1), color=color))
        return out

    def _render_current_sample(self) -> None:
        if not self.samples or self.current_index < 0:
            return

        payload = self.samples[self.current_index]
        image = payload["image"]
        gt_boxes = payload.get("ground_truth_boxes", [])
        pred_boxes = payload.get("predicted_boxes", [])
        prediction = payload.get("prediction", "")
        valid_labels = payload.get("valid_labels", [])
        is_correct = payload.get("correct", False)
        index = payload.get("index", "?")
        total = payload.get("total", "?")

        rendered = self._render_image(image=image, gt_boxes=gt_boxes, pred_boxes=pred_boxes)
        self._photo = ImageTk.PhotoImage(rendered)
        self.image_label.configure(image=self._photo)

        gt_labels_text = ", ".join(valid_labels) if valid_labels else "(none)"
        lines = [
            f"Sample: {index}/{total}",
            f"Prediction: {prediction}",
            f"Correct: {is_correct}",
            "",
            "Ground-truth labels:",
            gt_labels_text,
            "",
            f"Ground-truth boxes: {len(gt_boxes)}",
            f"Predicted boxes: {len(pred_boxes)}",
            "",
            "Legend:",
            "Green = ground truth box",
            "Orange = predicted box",
        ]
        self.meta_var.set("\n".join(lines))

        self.nav_var.set(f"Viewing saved sample {self.current_index + 1}/{len(self.samples)}")
        self.prev_button.configure(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL if self.current_index < len(self.samples) - 1 else tk.DISABLED)

    def _bring_to_front(self) -> None:
        if not self.enabled or self.root is None:
            return
        try:
            self.root.state("normal")
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.attributes("-topmost", False)
            self.root.focus_force()
        except tk.TclError:
            pass
