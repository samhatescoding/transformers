from __future__ import annotations

from .hf_common import HFQADataset


class VQAv2(HFQADataset):
    def __init__(self, split: str = "validation", streaming: bool = True, dataset_id: str = "merve/vqav2-small") -> None:
        actual_split = "validation" if split.startswith("val") else split
        super().__init__(
            name="vqav2",
            dataset_id=dataset_id,
            split=actual_split,
            streaming=streaming,
            answer_keys=("multiple_choice_answer", "answer", "answers"),
        )
