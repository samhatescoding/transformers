from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._base_model import BaseModel


class Orion14B(BaseModel):
    def __init__(
        self,
        model_id: str = "OrionStarAI/Orion-14B-Chat",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.name = "orion14b"
        self.model_id = "OrionStarAI/Orion-14B-Chat" if model_id == "orion14b" else model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print("Loading Orion14B tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Loading Orion14B model...")
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            self.model.to("cpu")

    @staticmethod
    def _adapt_prompt(prompt: str) -> str:
        p = prompt.replace("<image>", "")
        p = p.replace("USER:", "").replace("ASSISTANT:", "").strip()
        return p or "Describe the input."

    def predict(self, image: Image.Image, prompt: str) -> str:
        del image

        prompt = self._adapt_prompt(prompt)
        input_device = next(self.model.parameters()).device

        if hasattr(self.tokenizer, "apply_chat_template"):
            model_inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(input_device)
            generate_kwargs = {"input_ids": model_inputs}
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt").to(input_device)
            generate_kwargs = dict(encoded)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**generate_kwargs, **gen_kwargs)

        if "input_ids" in generate_kwargs:
            input_len = generate_kwargs["input_ids"].shape[-1]
            generated = output[0][input_len:]
        else:
            generated = output[0]

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
