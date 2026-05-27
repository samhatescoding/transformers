from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelRun:
    name: str
    model: object
    load_time_seconds: float | None = None

    @classmethod
    def from_factory(
        cls,
        name: str,
        factory: Callable[..., object],
        *args,
        **kwargs,
    ) -> "ModelRun":
        started_at = time.perf_counter()
        model = factory(*args, **kwargs)
        return cls(
            name=name,
            model=model,
            load_time_seconds=time.perf_counter() - started_at,
        )
