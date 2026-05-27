from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path


class FineTuningPreparationTests(unittest.TestCase):
    def test_fashion_mnist_balanced_indices_select_equal_classes_without_overlap(self) -> None:
        directory = Path(__file__).resolve().parents[1] / "fine-tuning"
        sys.path.insert(0, str(directory))
        try:
            module = runpy.run_path(str(directory / "prepare_fashion_mnist.py"))
        finally:
            sys.path.remove(str(directory))
        select_balanced_indices = module["select_balanced_indices"]
        rows = [{"label": label} for label in range(3) for _ in range(6)]

        train_indices, validation_indices = select_balanced_indices(
            rows,
            label_count=3,
            train_per_class=2,
            validation_per_class=1,
            seed=42,
        )

        self.assertEqual(len(train_indices), 6)
        self.assertEqual(len(validation_indices), 3)
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(
            {label: sum(rows[index]["label"] == label for index in train_indices) for label in range(3)},
            {0: 2, 1: 2, 2: 2},
        )
        self.assertEqual(
            {label: sum(rows[index]["label"] == label for index in validation_indices) for label in range(3)},
            {0: 1, 1: 1, 2: 1},
        )


if __name__ == "__main__":
    unittest.main()
