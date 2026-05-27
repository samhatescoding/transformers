"""QLoRA fine-tune Qwen2.5-VL-3B-Instruct on prepared Fashion-MNIST examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.root = manifest_path.parent
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        record["image"] = Image.open(self.root / record["image_path"]).convert("RGB")
        return record


class QwenVisionCollator:
    def __init__(self, processor: Any, system_prompt: str) -> None:
        self.processor = processor
        self.system_prompt = system_prompt

    def _conversation(self, example: dict[str, Any], include_answer: bool) -> list[dict]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["prompt"]},
                ],
            },
        ]
        if include_answer:
            messages.append({"role": "assistant", "content": example["answer"]})
        return messages

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_text = [
            self.processor.apply_chat_template(
                self._conversation(example, include_answer=True),
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]
        prompt_text = [
            self.processor.apply_chat_template(
                self._conversation(example, include_answer=False),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in examples
        ]
        images = [example["image"] for example in examples]
        batch = self.processor(
            text=full_text, images=images, padding=True, return_tensors="pt"
        )
        prompt_batch = self.processor(
            text=prompt_text, images=images, padding=True, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in zip(labels, prompt_lengths):
            row[: int(prompt_length)] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("fine-tuning/data/fashion_mnist/train_manifest.jsonl"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("fine-tuning/data/fashion_mnist/validation_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine-tuning/output/qwen2.5-vl-3b-fashion-mnist-lora"),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "Qwen2.5-VL-3B QLoRA training requires a CUDA GPU. "
            "Run this script in a GPU Colab runtime."
        )
    for manifest_path in (args.train_manifest, args.validation_manifest):
        if not manifest_path.is_file():
            raise SystemExit(
                f"Missing {manifest_path}. Run fine-tuning/prepare_fashion_mnist.py first."
            )

    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install Qwen fine-tuning dependencies: "
            "pip install -r fine-tuning/requirements-qwen.txt"
        ) from exc

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    system_prompt = (
        "Classify Fashion-MNIST clothing images using exactly one allowed label."
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=0.03,
            logging_steps=5,
            logging_first_step=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=False,
            fp16=compute_dtype == torch.float16,
            bf16=compute_dtype == torch.bfloat16,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=ManifestDataset(args.train_manifest),
        eval_dataset=ManifestDataset(args.validation_manifest),
        data_collator=QwenVisionCollator(processor, system_prompt),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
