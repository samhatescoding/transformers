from __future__ import annotations

from dataclasses import dataclass

from benchmarks._base_benchmark import BaseBenchmark


@dataclass(frozen=True)
class BenchmarkRun:
    benchmark: BaseBenchmark
    num_samples: int

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
