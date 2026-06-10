from dataset import TAD66K

from ..aesthetic_rating import AestheticRatingBenchmark


class TAD66KBenchmark(AestheticRatingBenchmark):
    dataset_cls = TAD66K
    benchmark_name = "tad66k"
    default_split = "train"

    def prepare(self, n: int, label_sample_size: int):
        sampler = getattr(self.dataset, "get_score_spaced_samples", None)
        if not callable(sampler):
            return super().prepare(n=n, label_sample_size=label_sample_size)
        rows = sampler(n)
        return rows, self.get_candidate_labels(rows)
