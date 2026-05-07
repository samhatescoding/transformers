from __future__ import annotations

from .hf_common import HFQADataset


class VisualGenome(HFQADataset):
    def __init__(self, split: str = "train", streaming: bool = True, dataset_id: str = "ranjaykrishna/visual_genome") -> None:
        super().__init__(
            name="visual_genome",
            dataset_id=dataset_id,
            split=split,
            streaming=streaming,
            question_keys=("question", "relationship_question"),
            answer_keys=("answer", "relationship", "triplet"),
        )
