from __future__ import annotations

import threading

import torch
from PIL import Image
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
    TextIteratorStreamer,
)

from ._hf_model import HuggingFaceModelBase


class LlavaOnevision(HuggingFaceModelBase):
    display_name = "LLaVA-OneVision"
    model_loader = LlavaOnevisionForConditionalGeneration.from_pretrained
    model_torch_dtype = torch.float16
    default_model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 64,
        stream: bool = False,
    ):
        self.model_id = model_id or self.default_model_id
        self.name = self._name_from_model_id(self.model_id)
        self.max_new_tokens = max_new_tokens
        self.stream = stream

        self._load_input_artifact_and_model()

    @staticmethod
    def _name_from_model_id(model_id: str) -> str:
        return str(model_id).rsplit("/", 1)[-1].lower().replace("_", "-")

    def _load_input_artifact(self):
        return self._load_with_cache_first(
            LlavaOnevisionProcessor.from_pretrained,
            self.model_id,
            "processor",
        )

    def predict(self, image: Image.Image, prompt: str) -> str:
        instruction = self._extract_prompt(prompt)
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
