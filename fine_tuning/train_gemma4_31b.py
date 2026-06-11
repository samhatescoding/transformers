"""Fine-tune google/gemma-4-31B-it with LoRA on image-text manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "google/gemma-4-31B-it"
DEFAULT_SYSTEM_PROMPT = (
    "Follow the visual task instructions and return only the requested answer format."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit"),
        default="none",
        help="Use BF16 LoRA by default. Select 4bit only if the BF16 run runs out of memory.",
    )
    parser.add_argument(
        "--train-vision-tower",
        action="store_true",
        help="Also train LoRA adapters in the vision tower. It is frozen by default.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="Resume from the latest checkpoint, or provide a checkpoint directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for manifest_path in (args.train_manifest, args.validation_manifest):
        if not manifest_path.is_file():
            raise SystemExit(
                f"Missing manifest: {manifest_path}\n"
                "Create it with fine_tuning/prepare_benchmark.py or provide a compatible JSONL file."
            )
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("Batch sizes must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be at least 1.")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise SystemExit("LoRA rank and alpha must be positive.")


def require_hugging_face_token() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise SystemExit(
            "Set HF_TOKEN after accepting the Gemma license at "
            "https://huggingface.co/google/gemma-4-31B-it"
        )


def load_records(manifest_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{manifest_path}:{line_number} is not valid JSON: {exc}"
            ) from exc
        missing = [
            key for key in ("image_path", "prompt", "answer") if not payload.get(key)
        ]
        if missing:
            raise ValueError(
                f"{manifest_path}:{line_number} is missing: {', '.join(missing)}"
            )
        image_path = (manifest_path.parent / payload["image_path"]).resolve()
        if not image_path.is_file():
            raise ValueError(
                f"{manifest_path}:{line_number} references missing image {image_path}"
            )
        records.append(
            {
                "image_path": str(image_path),
                "prompt": str(payload["prompt"]).strip(),
                "answer": str(payload["answer"]).strip(),
                "system": str(
                    payload.get("system") or DEFAULT_SYSTEM_PROMPT
                ).strip(),
            }
        )
    if not records:
        raise ValueError(f"{manifest_path} contains no training examples.")
    return records


def freeze_vision_adapter_parameters(model: Any) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "vision" in lowered or "visual" in lowered:
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()
    return frozen


def assert_trainable_parameters(model: Any) -> None:
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("LoRA produced no trainable parameters.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_hugging_face_token()

    try:
        import torch
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from PIL import Image
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies with: pip install -r fine_tuning/requirements.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("Gemma 4 31B fine-tuning requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This workflow requires a BF16-capable GPU.")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    class ManifestDataset(Dataset):
        def __init__(self, manifest_path: Path) -> None:
            self.records = load_records(manifest_path)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = dict(self.records[index])
            with Image.open(record["image_path"]) as image:
                record["image"] = image.convert("RGB")
            return record

    class GemmaVisionCollator:
        def __init__(self, processor: Any) -> None:
            self.processor = processor

        @staticmethod
        def messages(example: dict[str, Any], include_answer: bool) -> list[dict]:
            conversation = [
                {"role": "system", "content": example["system"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["prompt"]},
                    ],
                },
            ]
            if include_answer:
                conversation.append(
                    {"role": "assistant", "content": example["answer"]}
                )
            return conversation

        def render(self, example: dict[str, Any], include_answer: bool) -> str:
            return self.processor.apply_chat_template(
                self.messages(example, include_answer),
                tokenize=False,
                add_generation_prompt=not include_answer,
                enable_thinking=False,
            )

        def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
            images = [example["image"] for example in examples]
            full_text = [self.render(example, True) for example in examples]
            prompt_text = [self.render(example, False) for example in examples]

            batch = self.processor(
                text=full_text,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            prompt_batch = self.processor(
                text=prompt_text,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
            for labels_row, prompt_length in zip(labels, prompt_lengths):
                labels_row[: int(prompt_length)] = -100
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
            return batch

    processor = AutoProcessor.from_pretrained(args.model_id, token=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": torch.cuda.current_device()},
        "attn_implementation": "sdpa",
        "token": True,
    }
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForMultimodalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    if not args.train_vision_tower:
        frozen = freeze_vision_adapter_parameters(model)
        print(f"Frozen {frozen:,} vision-tower adapter parameters.")
    assert_trainable_parameters(model)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(args.train_manifest),
        eval_dataset=ManifestDataset(args.validation_manifest),
        data_collator=GemmaVisionCollator(processor),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved Gemma 4 31B LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
