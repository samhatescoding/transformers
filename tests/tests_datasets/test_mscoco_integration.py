from __future__ import annotations

import unittest

from PIL import Image

from benchmarks import MSCOCOBenchmark


class _RecordingModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, image: Image.Image, prompt: str) -> str:
        self.calls.append(
            {
                "mode": image.mode,
                "size": image.size,
                "prompt": prompt,
            }
        )
        return ""


class MSCOCOIntegrationTests(unittest.TestCase):
    def test_benchmark_passes_decoded_mscoco_image_to_model(self) -> None:
        benchmark = MSCOCOBenchmark(streaming=True)
        model = _RecordingModel()

        report = benchmark.run(
            model=model,
            n=1,
            label_sample_size=1,
            show_progress=False,
        )

        self.assertEqual(report["benchmark"], "mscoco")
        self.assertEqual(report["dataset"], "mscoco")
        self.assertEqual(report["num_samples"], 1)
        self.assertEqual(len(model.calls), 1)

        call = model.calls[0]
        self.assertEqual(call["mode"], "RGB")
        self.assertIsInstance(call["size"], tuple)
        self.assertEqual(len(call["size"]), 2)
        self.assertGreater(call["size"][0], 0)
        self.assertGreater(call["size"][1], 0)
        self.assertIn("USER: <image>", call["prompt"])


if __name__ == "__main__":
    unittest.main()
