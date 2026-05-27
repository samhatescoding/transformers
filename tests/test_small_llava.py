from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from models.llava15_7b import Llava


class _FakeBatch(dict):
    def to(self, device):
        del device
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.num_additional_image_tokens = 0
        self.calls = []
        self.chat_template_calls = []
        self.tokenizer = SimpleNamespace(
            chat_template="{{ bos_token }}<start_of_turn>",
            bos_token="<bos>",
        )

    def __call__(self, *, text, images, return_tensors):
        del text
        del images
        del return_tensors
        image_token_count = 575 + int(self.num_additional_image_tokens)
        input_ids = torch.tensor([[42] + [99] * image_token_count + [7]], dtype=torch.long)
        self.calls.append(image_token_count)
        return _FakeBatch({"input_ids": input_ids})

    def apply_chat_template(self, conversation, add_generation_prompt):
        self.chat_template_calls.append(
            {
                "conversation": conversation,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "<formatted>"


class SmallLlavaTests(unittest.TestCase):
    def test_repairs_missing_image_placeholder_token_before_generation(self) -> None:
        model = Llava.__new__(Llava)
        model.processor = _FakeProcessor()
        model.model = SimpleNamespace(
            device="cpu",
            config=SimpleNamespace(
                image_token_index=99,
                vision_config=SimpleNamespace(
                    patch_size=14,
                    image_size=336,
                ),
            ),
        )

        image = Image.new("RGB", (32, 32), "white")
        initial_inputs = model.processor(text="USER: <image>\nASSISTANT:", images=image, return_tensors="pt")
        repaired_inputs = model._repair_image_token_mismatch(
            prompt="USER: <image>\nASSISTANT:",
            image=image,
            inputs=initial_inputs,
        )

        self.assertEqual(model.processor.calls, [575, 576])
        self.assertEqual(model.processor.num_additional_image_tokens, 1)
        image_token_count = int((repaired_inputs["input_ids"][0] == 99).sum().item())
        self.assertEqual(image_token_count, 576)

    def test_cleans_duplicate_and_numeric_artifacts_from_generation(self) -> None:
        model = Llava.__new__(Llava)
        cleaned = model._clean_output_text("0\n0\nParakeet\nParakeet\n")
        self.assertEqual(cleaned, "Parakeet")

    def test_converts_legacy_prompt_to_multimodal_chat_template(self) -> None:
        model = Llava.__new__(Llava)
        model.processor = _FakeProcessor()
        formatted = model._prepare_prompt("USER: <image>\nAnswer briefly.\nASSISTANT:")

        self.assertEqual(
            formatted,
            "<bos><start_of_turn>user\n<image>\nAnswer briefly.<end_of_turn>\n<start_of_turn>model\n",
        )
        self.assertEqual(model.processor.chat_template_calls, [])

    def test_caption_cleanup_keeps_first_sentence(self) -> None:
        model = Llava.__new__(Llava)
        cleaned = model._clean_output_text(
            "The man is sitting on a motorcycle.\n0\nThe motorcycle",
            prompt="USER: <image>\nWrite one concise caption that describes the image.\nReturn only the caption text.\nASSISTANT:",
        )
        self.assertEqual(cleaned, "The man is sitting on a motorcycle.")

    def test_detection_cleanup_wraps_bare_box_coordinates(self) -> None:
        model = Llava.__new__(Llava)
        cleaned = model._clean_output_text(
            "bottle: 0.1, 0.2, 0.3, 0.4",
            prompt="USER: <image>\nDetect the objects in the image.\nOnly use labels from this list: bottle, cup\nASSISTANT:",
        )
        self.assertEqual(cleaned, "bottle: [0.1, 0.2, 0.3, 0.4]")

    def test_detection_cleanup_converts_sentence_mentions_to_full_image_boxes(self) -> None:
        model = Llava.__new__(Llava)
        cleaned = model._clean_output_text(
            "A man wearing a yellow helmet is climbing a cliff face.",
            prompt="USER: <image>\nDetect the objects in the image.\nOnly use labels from this list: A man, a yellow helmet, a cliff face, snow\nASSISTANT:",
        )
        self.assertIn("A man: [0.0, 0.0, 1.0, 1.0]", cleaned)
        self.assertIn("a yellow helmet: [0.0, 0.0, 1.0, 1.0]", cleaned)


if __name__ == "__main__":
    unittest.main()
