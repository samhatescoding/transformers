from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.results_browser import BenchmarkResultsBrowser, DEFAULT_RESULTS_DIR


def main() -> int:
    results_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    BenchmarkResultsBrowser(results_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
