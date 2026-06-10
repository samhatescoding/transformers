#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:-flickr30k}"
DATA_DIR="${DATA_DIR:-fine_tuning/data/${BENCHMARK}}"
OUTPUT_DIR="${OUTPUT_DIR:-fine_tuning/output/gemma-4-31b-${BENCHMARK}-lora}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-1000}"
VALIDATION_EXAMPLES="${VALIDATION_EXAMPLES:-200}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Accept the Gemma license and export the token first." >&2
  exit 1
fi

if [[ ! -f "${DATA_DIR}/train_manifest.jsonl" || ! -f "${DATA_DIR}/validation_manifest.jsonl" ]]; then
  python fine-tuning/prepare_benchmark.py \
    --benchmark "${BENCHMARK}" \
    --train-split "${TRAIN_SPLIT}" \
    --train-examples "${TRAIN_EXAMPLES}" \
    --validation-examples "${VALIDATION_EXAMPLES}" \
    --output-dir "${DATA_DIR}"
fi

python fine_tuning/train_gemma4_31b.py \
  --train-manifest "${DATA_DIR}/train_manifest.jsonl" \
  --validation-manifest "${DATA_DIR}/validation_manifest.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  "${@:2}"
