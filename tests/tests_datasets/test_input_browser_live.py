from __future__ import annotations

import os
import unittest

from PIL import Image

from ui.input_browser import (
    BENCHMARK_SPECS,
    BenchmarkInputBrowser,
    BenchmarkInputService,
)


RUN_LIVE_TESTS = os.getenv("RUN_INPUT_BROWSER_LIVE_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
BENCHMARK_FILTER = os.getenv("INPUT_BROWSER_LIVE_BENCHMARK", "").strip().casefold()


@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "Set RUN_INPUT_BROWSER_LIVE_TESTS=1 to load every real benchmark input.",
)
class LiveInputBrowserTests(unittest.TestCase):
    def test_every_real_benchmark_square_builds_an_image_and_prompt(self):
        service = BenchmarkInputService(sample_count=1, label_sample_size=1, streaming=True)
        browser = BenchmarkInputBrowser.__new__(BenchmarkInputBrowser)
        failures: list[str] = []
        tested = 0

        for spec in BENCHMARK_SPECS:
            if BENCHMARK_FILTER and BENCHMARK_FILTER not in spec.name.casefold():
                continue

            preview_holder = []

            def open_preview(selected_spec, row_index):
                preview_holder.append(service.preview(selected_spec, row_index))

            browser._show_preview = open_preview
            try:
                browser._benchmark_click_command(spec)()
                self.assertEqual(len(preview_holder), 1)
                preview = preview_holder[0]
                self.assertIsInstance(preview.image, Image.Image)
                self.assertEqual(preview.image.mode, "RGB")
                self.assertGreater(preview.image.width, 0)
                self.assertGreater(preview.image.height, 0)
                self.assertTrue(preview.prompt.strip())
                self.assertIn("<image>", preview.prompt)
                self.assertEqual(preview.row_index, 0)
                tested += 1
            except Exception as exc:
                failures.append(
                    f"{spec.type_code} / {spec.name}: {type(exc).__name__}: {exc}"
                )

        if failures:
            self.fail(
                "Some benchmark tiles could not build their real model input:\n"
                + "\n".join(failures)
            )
        self.assertGreater(tested, 0, "The benchmark filter selected no benchmarks.")


if __name__ == "__main__":
    unittest.main()
