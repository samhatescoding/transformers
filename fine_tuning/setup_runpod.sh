#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r fine_tuning/requirements.txt

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {name}")
print(f"VRAM: {memory_gib:.1f} GiB")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
PY
