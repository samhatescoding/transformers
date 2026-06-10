# Fine-Tuning

All fine-tuning utilities now live in this directory.

## Gemma 4 31B on RunPod

The main entry point is:

```text
fine_tuning/gemma4_31b_eight_types.ipynb
```

Open that notebook from a clone stored on persistent RunPod storage and run
all cells. It:

1. Installs the required packages.
2. Requests a Hugging Face token.
3. Exports balanced training and validation slices for eight task types.
4. Combines and shuffles those slices.
5. Fine-tunes `google/gemma-4-31B-it` with BF16 LoRA.
6. Saves checkpoints and the final adapter under
   `fine_tuning/output/gemma-4-31b-eight-types-lora`.

The selected datasets are:

| Type | Task | Benchmark |
|---|---|---|
| L | Labeling | Fashion-MNIST |
| A | Visual question answering | DocVQA |
| B | Bounding-box detection | Open Images V4 |
| C | Captioning | MS COCO captions |
| E | Editing prompt reconstruction | MagicBrush |
| G | Generation prompt reconstruction | DiffusionDB |
| P | Preference | Pick-a-Pic |
| R | Rating | TAD66K |

## LLaVA-OneVision 7B Eight-Type Workflow

Use these notebooks as a matched before/after pair:

```text
fine_tuning/llava_onevision_qwen2_7b_eight_types_baseline.ipynb
fine_tuning/llava_onevision_qwen2_7b_eight_types_finetuned.ipynb
```

The first evaluates the unmodified
`llava-hf/llava-onevision-qwen2-7b-ov-hf` checkpoint. The second exports a
balanced mixture for the same eight task types, trains a 4-bit QLoRA adapter
with `train_llava_onevision_7b.py`, and evaluates the adapter with the same
sample counts and split configuration. The mixture uses 980 training and 200
validation examples per type because the configured DocVQA validation mirror
contains 1,286 usable images and 100 are reserved for evaluation.

Before running, accept the Gemma license:
<https://huggingface.co/google/gemma-4-31B-it>.

The configured DocVQA mirror has no `train` split. Its `validation` split is
used as a source pool and divided into disjoint fine-tuning training and
validation slices. Do not use the remaining default DocVQA validation stream
as a final evaluation set; reserve `test` for evaluation.

The pod should have at least 150 GB of persistent disk. Dataset availability
and gated access can change, so an inaccessible dataset will stop its export
cell rather than silently omit a task type.

## Command-Line Gemma Workflow

For a single benchmark:

```bash
export HF_TOKEN=hf_your_token
bash fine_tuning/setup_runpod.sh
bash fine_tuning/run_finetuning.sh flickr30k
```

The underlying trainer is `train_gemma4_31b.py`. It uses BF16 LoRA by default,
freezes the vision tower, and masks prompt tokens from the loss. Use
`--quantization 4bit` only if BF16 training runs out of memory.

## Data Format

`prepare_benchmark.py` exports JSONL records with paths relative to the
manifest:

```json
{"image_path":"images/00001.png","prompt":"Describe the image.","answer":"A red car.","system":"Return only the answer."}
```

Custom manifests can be passed directly:

```bash
python fine_tuning/train_gemma4_31b.py \
  --train-manifest /workspace/data/train_manifest.jsonl \
  --validation-manifest /workspace/data/validation_manifest.jsonl \
  --output-dir /workspace/output/gemma4-lora
```

Keep final evaluation data separate from both manifests.

## Other Workflows

The directory also retains the existing:

- Qwen2.5-VL benchmark and Fashion-MNIST QLoRA trainers.
- GPT-4o vision fine-tuning JSONL builder and submission tools.
- Base and adapter evaluation scripts.

Use `python fine_tuning/prepare_benchmark.py --help` to list supported
repository benchmarks.
