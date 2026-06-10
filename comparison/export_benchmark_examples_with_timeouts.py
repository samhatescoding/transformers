from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.input_browser import BENCHMARK_SPECS

SKIP_BENCHMARKS = {"visual genome"}


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_examples"))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    for index, spec in enumerate(BENCHMARK_SPECS, start=1):
        if spec.name.casefold() in SKIP_BENCHMARKS:
            print(f"[{index}/{len(BENCHMARK_SPECS)}] {spec.type_code} / {spec.name} [EXPLICIT SKIP]", flush=True)
            continue
        entries = _read_manifest(manifest_path)
        successful = {
            (str(entry.get("type")), str(entry.get("name")))
            for entry in entries
            if entry.get("status") == "ok"
        }
        key = (spec.type_code, spec.name)
        if key in successful:
            print(f"[{index}/{len(BENCHMARK_SPECS)}] {spec.type_code} / {spec.name} [SKIP]", flush=True)
            continue

        command = [
            sys.executable,
            str(REPO_ROOT / "comparison" / "export_benchmark_examples.py"),
            "--output-dir",
            str(output_dir),
            "--resume",
            "--type",
            spec.type_code,
            "--benchmark",
            spec.name,
        ]
        print(f"[{index}/{len(BENCHMARK_SPECS)}] {spec.type_code} / {spec.name}", flush=True)
        try:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] exceeded {args.timeout_seconds} seconds", flush=True)
            entries = [
                entry
                for entry in _read_manifest(manifest_path)
                if (entry.get("type"), entry.get("name")) != key
            ]
            entries.append(
                {
                    "type": spec.type_code,
                    "name": spec.name,
                    "module": spec.module,
                    "class": spec.class_name,
                    "status": "skipped_timeout",
                    "error": f"Exceeded {args.timeout_seconds} seconds.",
                }
            )
            _write_manifest(manifest_path, entries)

    entries = _read_manifest(manifest_path)
    completed = sum(entry.get("status") == "ok" for entry in entries)
    skipped = [entry for entry in entries if entry.get("status") != "ok"]
    print(f"Completed {completed}/{len(BENCHMARK_SPECS)} exact examples.", flush=True)
    for entry in skipped:
        print(
            f"- {entry.get('type')} / {entry.get('name')}: "
            f"{entry.get('status')} ({entry.get('error', '')})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
