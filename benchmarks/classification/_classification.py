from __future__ import annotations

from PIL import Image

from .._base_benchmark import BaseBenchmark
from dataset._base_dataset import BaseDataset


class ClassificationBenchmark(BaseBenchmark):
    max_edge = 336

    def __init__(self, dataset: BaseDataset, name: str):
        super().__init__(dataset=dataset, name=name)

    def get_image_for_row(self, row):
        image = self.dataset.get_image_from_row(row)
        if not isinstance(image, Image.Image):
            return image
        return self._resize_image(image)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        largest_edge = max(width, height)
        if largest_edge <= self.max_edge:
            return image
        scale = self.max_edge / float(largest_edge)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size, Image.Resampling.BICUBIC)
