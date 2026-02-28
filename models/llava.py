# models/llava.py

import threading
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TextIteratorStreamer,
)

from PIL import Image
from .base import BaseModel


class Llava(BaseModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        max_new_tokens: int = 50,
        stream: bool = True,
    ):
        self.name = "llava"
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.stream = stream

        print("Loading LLaVA processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        print("Loading LLaVA model...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        """
        try:
            print("Device map:", getattr(self.model, "hf_device_map", None))
        except Exception:
            pass
        """

    def predict(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        if self.stream:
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )

            #print("Generating: ", end="", flush=True)

            thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_kwargs
            )
            thread.start()

            output_text = ""
            for piece in streamer:
                #print(piece, end="", flush=True)
                output_text += piece

            thread.join()
            print()
        else:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated = output[0][inputs["input_ids"].shape[-1]:]
            output_text = self.processor.decode(
                generated,
                skip_special_tokens=True
            )

        return output_text.strip()
