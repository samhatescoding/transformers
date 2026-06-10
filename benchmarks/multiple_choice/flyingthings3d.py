from dataset import FlyingThings3D

from ..curated_answer_choices import get_curated_answer_row
from ._multiple_choice import MultipleChoiceBenchmark


class FlyingThings3DBenchmark(MultipleChoiceBenchmark):
    dataset_cls = FlyingThings3D
    benchmark_name = "flyingthings3d"
    default_split = "train"
    default_instruction = "Choose the description that best matches the rendered stereo scene."
    fallback_distractors = ("a close-up face video", "a document page", "a city street segmentation")

    def prepare(self, n, label_sample_size):
        rows = self.dataset.get_samples(max(n, label_sample_size))
        prepared = []
        for row_index, source_row in enumerate(rows[:n]):
            row = dict(source_row)
            curated = get_curated_answer_row(self.dataset.name, row_index)
            if curated is not None:
                source_id = str(row.get("id", "")).strip()
                curated_source_id = str(curated.get("source_id", "")).strip()
                if not curated_source_id or source_id == curated_source_id:
                    row["question"] = str(curated["question"])
                    row["answer"] = str(curated["answer"])
                    row["choices"] = list(curated["choices"])
            prepared.append(self._ensure_choices(row, rows))
        return prepared, []
