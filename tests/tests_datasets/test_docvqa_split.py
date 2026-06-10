from __future__ import annotations

import unittest
from unittest.mock import patch

from dataset.docvqa import DocVQA


class DocVQASplitTests(unittest.TestCase):
    @patch("dataset.hf_common.load_dataset")
    def test_train_alias_uses_available_validation_split(self, load_dataset) -> None:
        load_dataset.return_value = []

        dataset = DocVQA(split="train")

        load_dataset.assert_called_once_with(
            "lmms-lab/DocVQA",
            "DocVQA",
            split="validation",
            streaming=True,
        )
        self.assertEqual(dataset.split, "validation")


if __name__ == "__main__":
    unittest.main()
