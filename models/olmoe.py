from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._base_model import BaseModel


class OLMoE(BaseModel):
    def __init__(
        self,
        model_id: str = "allenai/OLMoE-1B-7B-0924",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        self.name = "olmoe"
        self.model_id = "allenai/OLMoE-1B-7B-0924" if model_id == "olmoe" else model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print("Loading OLMoE tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Loading OLMoE model...")
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(input_device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
