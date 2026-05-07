from __future__ import annotations

import json
from pathlib import Path


SAMPLE_STAT_KEYS_TO_REMOVE = {
    "invalid_format",
    "empty_response",
    "parse_failure",
}

RUN_STAT_KEYS_TO_REMOVE = {
    "invalid_format_rate",
    "empty_response_rate",
    "parse_failure_rate",
}


def scrub_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload.get("report")
    if not isinstance(report, dict):
        return

    for result in report.get("results", []):
        stats = result.get("stats")
        if isinstance(stats, dict):
            for key in SAMPLE_STAT_KEYS_TO_REMOVE:
                stats.pop(key, None)

    run_stats = report.get("stats")
    if isinstance(run_stats, dict):
        for key in RUN_STAT_KEYS_TO_REMOVE:
            run_stats.pop(key, None)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    for path in sorted(Path("results").glob("*.json")):
        scrub_file(path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
