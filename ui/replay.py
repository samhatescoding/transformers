from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .viewer import BenchmarkUI


def _load_saved_samples(run_dir: Path) -> List[Dict[str, Any]]:
    sample_paths = sorted(run_dir.glob("sample_*.json"))
    samples: List[Dict[str, Any]] = []
    for sample_path in sample_paths:
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        image_name = payload.get("image_file")
        if not isinstance(image_name, str) or not image_name.strip():
            continue

        image_path = run_dir / image_name
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        payload["image"] = image
        payload["pre_rendered_image"] = True
        samples.append(payload)
    return samples


def visualize_saved_ui_outputs(
    run_dir: str | Path | None = None,
    base_dir: str | Path = "ui/ui_outputs",
    title: str = "Saved Benchmark Viewer",
) -> BenchmarkUI:
    base_path = Path(base_dir)
    target_dir = Path(run_dir) if run_dir is not None else _find_latest_run_dir(base_path)
    if target_dir is None or not target_dir.exists():
        raise FileNotFoundError(f"No saved UI output run found in {base_path}")

    viewer = BenchmarkUI(title=title)
    if not viewer.enabled:
        return viewer

    samples = _load_saved_samples(target_dir)
    if not samples:
        viewer.close()
        raise FileNotFoundError(f"No saved sample JSON files were found in {target_dir}")

    viewer.samples.extend(samples)
    viewer.current_index = 0
    viewer._render_current_sample()
    viewer._bring_to_front()
    viewer.root.update_idletasks()
    viewer.root.update()
    viewer.wait_until_closed()
    return viewer


def _find_latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    run_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)
