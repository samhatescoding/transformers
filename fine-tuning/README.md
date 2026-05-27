# GPT-4o Vision Fine-Tuning

This directory contains a small workflow for supervised vision fine-tuning of
`gpt-4o-2024-08-06`. It is intended for tasks where each example includes an
image, an instruction, and the desired text answer, such as domain-specific
image classification.

OpenAI documents vision fine-tuning with `gpt-4o-2024-08-06`, but availability
is restricted. Its deprecations page states that, beginning May 7, 2026,
organizations that had not previously run fine-tuning cannot create training
jobs. Beginning July 2, 2026, further restrictions apply to organizations
that have not run inference on a fine-tuned model in the preceding 60 days.
An API `403` response with code `training_not_available` means the current
organization is not eligible to create the job.

Do not build training data from images or answers used in this repository's
benchmark evaluation results. Training on evaluation samples invalidates the
benchmark comparison. Create a separate training set and keep a held-out
validation and evaluation set.

## Security

An API key was previously pasted into notebook code. Revoke that key in the
OpenAI dashboard and create a replacement before running submission commands.
Set the replacement as `OPENAI_API_KEY` in the environment or Colab Secrets;
never put it into a notebook or commit it to git.

## Input Manifest

Prepare one JSONL manifest line for each supervised example. Each line must
contain `prompt`, `answer`, and exactly one of `image_path` or `image_url`.
Optional fields are `system` and `detail` (`low`, `high`, or `auto`).

Local image example:

```json
{"image_path":"training-data/item_001.png","prompt":"Return exactly one category name: damaged, intact.","answer":"damaged","detail":"low"}
```

Hosted image example:

```json
{"image_url":"https://example.org/images/item_002.jpg","prompt":"Return exactly one category name: damaged, intact.","answer":"intact","detail":"low"}
```

Images must be JPEG, PNG, or WEBP in RGB or RGBA mode and no larger than
10 MB. Per OpenAI's current documentation, training images with people,
faces, children, or CAPTCHAs are excluded. Local image paths are encoded into
Base64 data URLs; remote URLs must be publicly accessible during training.

## Prepare Data

Create training and validation files from separate manifests:

```powershell
python fine-tuning\build_vision_jsonl.py `
  --manifest fine-tuning\data\train_manifest.jsonl `
  --output fine-tuning\data\train_openai.jsonl `
  --default-system "Classify the image according to the requested labels."

python fine-tuning\build_vision_jsonl.py `
  --manifest fine-tuning\data\validation_manifest.jsonl `
  --output fine-tuning\data\validation_openai.jsonl `
  --default-system "Classify the image according to the requested labels."
```

Use `--image-base-dir` if manifest image paths are relative to a directory
other than the repository root.

## Submit Training

Submission uploads files and creates a paid fine-tuning job. The script will
not submit unless `--confirm-submit` is provided:

```powershell
$env:OPENAI_API_KEY = "your-rotated-key"
python fine-tuning\submit_vision_fine_tune.py `
  --training-file fine-tuning\data\train_openai.jsonl `
  --validation-file fine-tuning\data\validation_openai.jsonl `
  --suffix image-label-task `
  --confirm-submit
```

Check an existing job:

```powershell
python fine-tuning\check_fine_tune.py ftjob-...
```

If submission is rejected with `training_not_available`, the script reports
the uploaded file IDs. The job was not created, but the uploaded training
files may remain in the OpenAI project and can be deleted from its Files
storage if no longer needed.

## Fashion-MNIST Pilot

Fashion-MNIST is suitable for a first vision fine-tune because it contains
clothing images rather than people or faces. The repository benchmark is
configured to evaluate it on its held-out `test` split.

Generate a balanced, reproducible dataset. This selects equal numbers of each
class from Fashion-MNIST `train`; it never writes test examples into
fine-tuning data:

```powershell
python fine-tuning\prepare_fashion_mnist.py `
  --train-per-class 30 `
  --validation-per-class 10 `
  --output-dir fine-tuning\data\fashion_mnist_balanced
```

After reviewing the generated JSONL files and setting a rotated API key,
submit the pilot:

```powershell
python fine-tuning\submit_vision_fine_tune.py `
  --training-file fine-tuning\data\fashion_mnist_balanced\train_openai.jsonl `
  --validation-file fine-tuning\data\fashion_mnist_balanced\validation_openai.jsonl `
  --suffix fashion-mnist-pilot `
  --confirm-submit
```

Evaluate the base model and, once training finishes, the fine-tuned model on
Fashion-MNIST `test`:

```powershell
python fine-tuning\evaluate_fashion_mnist.py --model-id gpt-4o --model-name gpt-4o-base
python fine-tuning\evaluate_fashion_mnist.py --model-id ft:gpt-4o-2024-08-06:... --model-name gpt-4o-fashion-mnist-ft
```

## Source

OpenAI vision fine-tuning documentation:
https://developers.openai.com/api/docs/guides/vision-fine-tuning

OpenAI fine-tuning availability update:
https://developers.openai.com/api/docs/deprecations#update-to-openais-self-serve-fine-tuning

## Qwen2.5-VL-3B Alternative

For organizations unable to create OpenAI fine-tuning jobs, the repository
supports a QLoRA fine-tune of the open-weight
`Qwen/Qwen2.5-VL-3B-Instruct` model. The adapter is trained using only
Fashion-MNIST `train` examples and evaluated on `test`.

This machine is not sufficient for the training run: it exposes CPU-only
PyTorch and a 4 GB GPU. Use a Google Colab GPU runtime, preferably an L4 or
A100. The pilot uses 4-bit NF4 loading plus LoRA to reduce GPU memory use.

In Colab, after cloning this repository:

```python
%cd /content/transformers
!pip install -r requirements.txt -r fine-tuning/requirements-qwen.txt
!pip uninstall -y torchao

!python fine-tuning/prepare_fashion_mnist.py \
    --train-per-class 30 \
    --validation-per-class 10 \
    --output-dir fine-tuning/data/fashion_mnist_balanced

!python fine-tuning/train_qwen25vl_fashion_mnist.py \
    --train-manifest fine-tuning/data/fashion_mnist_balanced/train_manifest.jsonl \
    --validation-manifest fine-tuning/data/fashion_mnist_balanced/validation_manifest.jsonl \
    --output-dir fine-tuning/output/qwen2.5-vl-3b-fashion-mnist-balanced-lora \
    --epochs 2 \
    --learning-rate 1e-4
```

Evaluate the original and adapted models on identical held-out examples:

```python
!python fine-tuning/evaluate_qwen25vl_fashion_mnist.py --samples 50

!python fine-tuning/evaluate_qwen25vl_fashion_mnist.py \
    --adapter-path fine-tuning/output/qwen2.5-vl-3b-fashion-mnist-balanced-lora \
    --samples 50
```

If adapted-model evaluation fails with an error that Colab's installed
`torchao` is incompatible with `peft`, uninstall `torchao` and rerun only the
evaluation command:

```python
!pip uninstall -y torchao

!python fine-tuning/evaluate_qwen25vl_fashion_mnist.py \
    --adapter-path fine-tuning/output/qwen2.5-vl-3b-fashion-mnist-balanced-lora \
    --samples 50
```

No retraining is needed after that import-time adapter loading failure.

The original 100-example pilot was imbalanced (`Shirt`: 16 examples and
`Trouser`: 3 examples) and produced no held-out accuracy gain after corrected
label normalization. The balanced second run is intentionally larger (300
training and 100 validation examples) and uses a lower learning rate to reduce
class collapse.

The adapter checkpoints are written under `fine-tuning/output/` and are
ignored by git. Copy the resulting evaluation JSON files from
`results/fine-tuning/` into version control only when you want to retain the
measured comparison.

References:
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- https://huggingface.co/docs/peft/developer_guides/quantization
