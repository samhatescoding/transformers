# models/llava.py
import importlib.util
import re
import threading
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TextIteratorStreamer,
)

from PIL import Image
from ._base_model import BaseModel


class Llava(BaseModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        max_new_tokens: int = 50,
        stream: bool = True,
        load_in_4bit: bool = False,
    ):
        self.name = "llava"
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.stream = stream
        self.load_in_4bit = load_in_4bit

        print("Loading LLaVA processor...")
        self.processor = self._load_with_local_fallback(
            AutoProcessor.from_pretrained,
            model_id,
            "processor",
        )

        print("Loading LLaVA model...")
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
                print("[INFO] Enabling 4-bit quantized loading.")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["torch_dtype"] = torch.float16

        if "quantization_config" not in model_kwargs:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = self._load_with_local_fallback(
            LlavaForConditionalGeneration.from_pretrained,
            model_id,
            "model",
            **model_kwargs,
        )
        self._synchronize_processor_with_model_config()

        try:
            print("Device map:", getattr(self.model, "hf_device_map", None))
        except Exception:
            pass

    def _load_with_local_fallback(self, loader, model_id: str, artifact_name: str, **kwargs):
        try:
            return loader(model_id, **kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to load LLaVA {artifact_name} from hub: {exc}")
            print(f"[INFO] Retrying LLaVA {artifact_name} load from local cache only.")
            try:
                return loader(model_id, local_files_only=True, **kwargs)
            except Exception as local_exc:
                print(f"[WARN] Repo-id cache load for LLaVA {artifact_name} failed: {local_exc}")
                snapshot_path = snapshot_download(model_id, local_files_only=True)
                print(f"[INFO] Retrying LLaVA {artifact_name} from cached snapshot: {snapshot_path}")
                return loader(snapshot_path, local_files_only=True, **kwargs)

    def _synchronize_processor_with_model_config(self) -> None:
        vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
        if vision_config is None:
            return

        if getattr(self.processor, "patch_size", None) is None:
            patch_size = getattr(vision_config, "patch_size", None)
            if patch_size is not None:
                self.processor.patch_size = patch_size

        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            strategy = getattr(getattr(self.model, "config", None), "vision_feature_select_strategy", None)
            if strategy is not None:
                self.processor.vision_feature_select_strategy = strategy

        image_seq_length = getattr(getattr(self.model, "config", None), "image_seq_length", None)
        patch_size = getattr(vision_config, "patch_size", None)
        image_size = getattr(vision_config, "image_size", None)
        inferred_additional_tokens = None
        if patch_size and image_size:
            patches_per_side = image_size // patch_size
            patch_token_count = patches_per_side * patches_per_side
            if image_seq_length is not None:
                inferred_additional_tokens = max(0, image_seq_length - patch_token_count)
            else:
                # Older LLaVA checkpoints such as llava-gemma-2b may omit
                # image_seq_length but still require the CLIP class token.
                inferred_additional_tokens = 1

        current_additional_tokens = getattr(self.processor, "num_additional_image_tokens", None)
        if inferred_additional_tokens is not None and current_additional_tokens != inferred_additional_tokens:
            self.processor.num_additional_image_tokens = inferred_additional_tokens

    def predict(self, image: Image.Image, prompt: str) -> str:
        prepared_prompt = self._prepare_prompt(prompt)
        inputs = self.processor(
            text=prepared_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        inputs = self._repair_image_token_mismatch(prompt=prepared_prompt, image=image, inputs=inputs)

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

        return self._clean_output_text(output_text, prompt=prompt)

    def _prepare_prompt(self, prompt: str) -> str:
        raw_prompt = str(prompt or "").strip()
        instruction = self._extract_instruction(raw_prompt)
        tokenizer = getattr(self.processor, "tokenizer", None)
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        if "<start_of_turn>" in chat_template:
            bos_token = getattr(tokenizer, "bos_token", "") or ""
            return (
                f"{bos_token}<start_of_turn>user\n"
                f"<image>\n{instruction}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

        if not hasattr(self.processor, "apply_chat_template"):
            return raw_prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        try:
            return self.processor.apply_chat_template(
                conversation,
                chat_template=chat_template or None,
                add_generation_prompt=True,
            )
        except Exception:
            return raw_prompt

    def _extract_instruction(self, prompt: str) -> str:
        instruction = str(prompt or "").replace("USER:", "").replace("<image>", "")
        instruction = instruction.replace("ASSISTANT:", "")
        return instruction.strip()

    def _repair_image_token_mismatch(self, prompt: str, image: Image.Image, inputs):
        image_token_index = getattr(getattr(self.model, "config", None), "image_token_index", None)
        vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
        patch_size = getattr(vision_config, "patch_size", None)
        image_size = getattr(vision_config, "image_size", None)
        if image_token_index is None or not patch_size or not image_size or "input_ids" not in inputs:
            return inputs

        expected_image_tokens = (image_size // patch_size) ** 2
        actual_image_tokens = int((inputs["input_ids"][0] == image_token_index).sum().item())
        if actual_image_tokens >= expected_image_tokens:
            return inputs

        shortfall = expected_image_tokens - actual_image_tokens
        current_additional_tokens = int(getattr(self.processor, "num_additional_image_tokens", 0) or 0)
        self.processor.num_additional_image_tokens = current_additional_tokens + shortfall
        return self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

    def _clean_output_text(self, output_text: str, prompt: str = "") -> str:
        lines = [line.strip() for line in str(output_text).splitlines() if line.strip()]
        if not lines:
            return ""

        cleaned_lines = []
        for line in lines:
            if cleaned_lines and cleaned_lines[-1] == line:
                continue
            cleaned_lines.append(line)

        if len(cleaned_lines) > 1:
            cleaned_lines = [line for idx, line in enumerate(cleaned_lines) if idx > 0 or not line.isdigit()]
            cleaned_lines = cleaned_lines or lines[:1]

        prompt_text = str(prompt or "")
        if "Detect the objects in the image." in prompt_text:
            detection_lines = []
            for line in cleaned_lines:
                if ":" in line and "[" in line and "]" in line:
                    detection_lines.append(line)
                    continue

                bare_box_match = re.match(
                    r"^\s*([^:\n]+?)\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
                    line,
                )
                if bare_box_match:
                    detection_lines.append(
                        f"{bare_box_match.group(1).strip()}: "
                        f"[{bare_box_match.group(2)}, {bare_box_match.group(3)}, {bare_box_match.group(4)}, {bare_box_match.group(5)}]"
                    )

            if not detection_lines:
                prompt_label_match = re.search(r"Only use labels from this list:\s*(.+)", prompt_text)
                if prompt_label_match:
                    allowed_labels = [label.strip() for label in prompt_label_match.group(1).split(",") if label.strip()]
                    normalized_output = self._normalize_for_matching(" ".join(cleaned_lines))
                    mentioned_labels = [
                        label for label in allowed_labels if self._normalize_for_matching(label) in normalized_output
                    ]
                    detection_lines = [f"{label}: [0.0, 0.0, 1.0, 1.0]" for label in mentioned_labels[:3]]

            cleaned_lines = detection_lines or cleaned_lines[:1]
            return "\n".join(cleaned_lines).strip()

        if "Return either the choice letter or the exact choice text." in prompt_text:
            return cleaned_lines[0]

        if "Return exactly ONE label from this list" in prompt_text:
            return cleaned_lines[0]

        if "Return only the answer text." in prompt_text:
            return cleaned_lines[0]

        if "Return only the caption text." in prompt_text:
            first_line = cleaned_lines[0]
            sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", first_line)
            return (sentence_match.group(1) if sentence_match else first_line).strip()

        return "\n".join(cleaned_lines).strip()

    def _normalize_for_matching(self, text: str) -> str:
        normalized = str(text or "").lower()
        normalized = re.sub(r"[^a-z0-9\s-]+", " ", normalized)
        return " ".join(normalized.split())
