# Gemma 4 31B Fine-Tuning

This directory provides a single-GPU BF16 LoRA workflow for
`google/gemma-4-31B-it` on a RunPod B300. It uses the repository's existing
benchmark exporter and trains only on assistant responses. The vision tower is
frozen by default; LoRA adapters are trained in the language model and
multimodal projection layers.

## RunPod setup

1. Create a B300 pod with at least 150 GB of persistent disk.
2. Accept the model license at:
   <https://huggingface.co/google/gemma-4-31B-it>
3. Clone this repository and run:

```bash
cd transformers
export HF_TOKEN=hf_your_token
bash fine_tuning/setup_runpod.sh
bash fine_tuning/run_finetuning.sh flickr30k
```

The launcher creates 1,000 training and 200 validation examples from the
requested benchmark if manifests do not already exist. It then writes the LoRA
adapter to:

```text
fine_tuning/output/gemma-4-31b-flickr30k-lora
```

Change the defaults with environment variables:

```bash
TRAIN_EXAMPLES=5000 \
VALIDATION_EXAMPLES=500 \
TRAIN_SPLIT=train \
OUTPUT_DIR=/workspace/gemma4-flickr30k-lora \
bash fine_tuning/run_finetuning.sh flickr30k \
  --epochs 2 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 16
```

Arguments after the benchmark name are passed to `train_gemma4_31b.py`.

## Supported data

The first launcher argument must be a benchmark accepted by:

```bash
python fine-tuning/prepare_benchmark.py --help
```

Examples include `flickr30k`, `docvqa`, `mscoco`,
`openimages_v4_detection`, `ucf101`, and `conceptual_captions`.

You can also provide your own JSONL manifests and run the trainer directly:

```bash
python fine_tuning/train_gemma4_31b.py \
  --train-manifest /workspace/data/train_manifest.jsonl \
  --validation-manifest /workspace/data/validation_manifest.jsonl \
  --output-dir /workspace/output/gemma4-lora
```

Each JSONL row must contain paths relative to the manifest:

```json
{"image_path":"images/00001.png","prompt":"Describe the image.","answer":"A red car.","system":"Return only the answer."}
```

Keep the final test split completely separate from the training and validation
manifests.

## Memory settings

The default is BF16 LoRA, which is appropriate for a 288 GB B300. If a
high-resolution dataset still runs out of memory, retry with:

```bash
bash fine_tuning/run_finetuning.sh docvqa --quantization 4bit
```

Other useful controls:

```text
--lora-rank 32
--batch-size 1
--gradient-accumulation-steps 16
--dataloader-workers 4
--resume-from-checkpoint
```

Do not enable `--train-vision-tower` for the first experiment. It increases
memory use and the risk of degrading general visual representations.

The output directory contains a PEFT adapter, not a second copy of the 62.6 GB
base model. Load it together with `google/gemma-4-31B-it` for evaluation.
