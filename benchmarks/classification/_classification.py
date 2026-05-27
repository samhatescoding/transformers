from __future__ import annotations

from PIL import Image

from .._base_benchmark import BaseBenchmark


class ClassificationBenchmark(BaseBenchmark):
    max_edge = 336

    def get_image_for_row(self, row):
        image = self.dataset.get_image_from_row(row)
        if not isinstance(image, Image.Image):
            return image
        return self._resize_image(image)
