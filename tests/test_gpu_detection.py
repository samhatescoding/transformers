from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmarks.resource_sampler import ResourceSampler


class ResourceSamplerGpuDetectionTests(unittest.TestCase):
    def test_records_gpu_peak(self) -> None:
        sampler = ResourceSampler(interval_seconds=0.001)
        with (
            patch("benchmarks.resource_sampler.torch.cuda.reset_peak_memory_stats") as reset_peak,
            patch("benchmarks.resource_sampler.torch.cuda.memory_allocated", return_value=128),
            patch("benchmarks.resource_sampler.torch.cuda.max_memory_allocated", return_value=256),
        ):
            sampler.start()
            stats = sampler.stop()

        reset_peak.assert_called_once()
        self.assertEqual(stats["peak_gpu_memory_bytes"], 256)
        self.assertIsNone(stats["gpu_utilization_percent"])
        self.assertIsNone(stats["vram_allocation_over_time_bytes"])


if __name__ == "__main__":
    unittest.main()
