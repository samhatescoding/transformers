from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_UI_OUTPUTS_DIR = ROOT_DIR / "ui" / "ui_outputs"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui import visualize_saved_ui_outputs


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    try:
        visualize_saved_ui_outputs(run_dir=run_dir, base_dir=DEFAULT_UI_OUTPUTS_DIR)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
