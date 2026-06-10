from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.input_browser import BENCHMARK_SPECS, BenchmarkInputService


def _slug(type_code: str, name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return f"{type_code.casefold()}_{normalized}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export exact benchmark images and prompts through BenchmarkInputService."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_examples"),
        help="Directory for PNG images and manifest.json.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Optional case-insensitive benchmark-name filter. May be repeated.",
    )
    parser.add_argument(
        "--type",
        action="append",
        default=[],
        help="Optional benchmark type-code filter. May be repeated.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="append",
        default=[],
        help="Optional case-insensitive benchmark name to skip. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful manifest entries and skip their benchmark names.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    filters = {value.strip().casefold() for value in args.benchmark if value.strip()}
    type_filters = {value.strip().upper() for value in args.type if value.strip()}
    skipped_names = {
        value.strip().casefold()
        for value in args.skip_benchmark
        if value.strip()
    }
    service = BenchmarkInputService(sample_count=1, label_sample_size=1, streaming=True)
    entries: list[dict[str, object]] = []
    if args.resume and manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    successful_keys = {
        (str(entry.get("type")), str(entry.get("name")))
        for entry in entries
        if entry.get("status") == "ok"
    }

    for index, spec in enumerate(BENCHMARK_SPECS, start=1):
        if filters and spec.name.casefold() not in filters:
            continue
        if type_filters and spec.type_code not in type_filters:
            continue
        if spec.name.casefold() in skipped_names:
            continue
        if (spec.type_code, spec.name) in successful_keys:
            print(f"[{index}/{len(BENCHMARK_SPECS)}] {spec.type_code} / {spec.name} [SKIP]", flush=True)
            continue

        slug = _slug(spec.type_code, spec.name)
        image_path = image_dir / f"{slug}.png"
        print(f"[{index}/{len(BENCHMARK_SPECS)}] {spec.type_code} / {spec.name}", flush=True)
        try:
            preview = service.preview(
                spec,
                0,
                progress=lambda step, total, message: print(
                    f"  [{step}/{total}] {message}",
                    flush=True,
                ),
            )
            if args.overwrite or not image_path.exists():
                preview.image.save(image_path, format="PNG")
            entry = {
                "type": spec.type_code,
                "name": spec.name,
                "module": spec.module,
                "class": spec.class_name,
                "benchmark_name": preview.benchmark_name,
                "dataset_name": preview.dataset_name,
                "source": preview.source,
                "split": preview.split,
                "row_index": preview.row_index,
                "row_count": preview.row_count,
                "width": preview.image.width,
                "height": preview.image.height,
                "image_path": image_path.relative_to(output_dir).as_posix(),
                "prompt": preview.prompt,
                "prompt_labels": preview.prompt_labels,
                "correct_answers": preview.correct_answers,
                "show_correct_answer": preview.show_correct_answer,
                "displayed_box_count": preview.displayed_box_count,
                "status": "ok",
            }
            print(f"  [OK] {image_path}", flush=True)
        except Exception as exc:
            entry = {
                "type": spec.type_code,
                "name": spec.name,
                "module": spec.module,
                "class": spec.class_name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"  [ERROR] {entry['error']}", flush=True)

        entries = [
            existing
            for existing in entries
            if (existing.get("type"), existing.get("name")) != (spec.type_code, spec.name)
        ]
        entries.append(entry)
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)

    failures = [entry for entry in entries if entry["status"] != "ok"]
    print(
        f"Exported {len(entries) - len(failures)}/{len(entries)} examples to {output_dir}",
        flush=True,
    )
    if failures:
        print("Failures:", flush=True)
        for entry in failures:
            print(f"- {entry['type']} / {entry['name']}: {entry['error']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
