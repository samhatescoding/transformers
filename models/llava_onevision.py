from __future__ import annotations

import importlib.util
import threading

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
    TextIteratorStreamer,
)

from ._base_model import BaseModel


class LlavaOnevision(BaseModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-72b-ov-hf",
        max_new_tokens: int = 64,
        stream: bool = False,
        load_in_4bit: bool = False,
    ):
        self.name = "llava-onevision-72b"
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.stream = stream
        self.load_in_4bit = load_in_4bit

        print("Loading LLaVA-OneVision processor...")
        self.processor = self._load_with_local_fallback(
            LlavaOnevisionProcessor.from_pretrained,
            model_id,
            "processor",
        )

        print("Loading LLaVA-OneVision model...")
        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            if not torch.cuda.is_available():
                print("[WARN] 4-bit loading requested, but CUDA is unavailable. Falling back to the standard loader.")
            elif importlib.util.find_spec("bitsandbytes") is None:
                print("[WARN] 4-bit loading requested, but bitsandbytes is not installed. Falling back to the standard loader.")
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["torch_dtype"] = torch.float16

        if "quantization_config" not in model_kwargs:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = self._load_with_local_fallback(
            LlavaOnevisionForConditionalGeneration.from_pretrained,
            model_id,
            "model",
            **model_kwargs,
        )

    def _load_with_local_fallback(self, loader, model_id: str, artifact_name: str, **kwargs):
        try:
            return loader(model_id, **kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to load LLaVA-OneVision {artifact_name} from hub: {exc}")
            print(f"[INFO] Retrying LLaVA-OneVision {artifact_name} load from local cache only.")
            try:
                return loader(model_id, local_files_only=True, **kwargs)
            except Exception as local_exc:
                print(f"[WARN] Repo-id cache load for LLaVA-OneVision {artifact_name} failed: {local_exc}")
                snapshot_path = snapshot_download(model_id, local_files_only=True)
                print(f"[INFO] Retrying LLaVA-OneVision {artifact_name} from cached snapshot: {snapshot_path}")
                return loader(snapshot_path, local_files_only=True, **kwargs)

    def predict(self, image: Image.Image, prompt: str) -> str:
        instruction = (
            str(prompt or "")
            .replace("USER:", "")
            .replace("ASSISTANT:", "")
            .replace("<image>", "")
            .strip()
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=rendered,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        if self.stream:
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            thread = threading.Thread(
                target=self.model.generate,
                kwargs={
                    **inputs,
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "streamer": streamer,
                },
            )
            thread.start()
            output_text = "".join(piece for piece in streamer)
            thread.join()
        else:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = output[0][inputs["input_ids"].shape[-1]:]
            output_text = self.processor.decode(generated, skip_special_tokens=True)

        return str(output_text).strip()
