from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import psutil
import torch


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_samples: List[float] = []
        self._rss_peak_bytes = 0
        self._gpu_memory_peak_bytes: int | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._rss_peak_bytes = self.process.memory_info().rss
        self.process.cpu_percent(interval=None)
        self._gpu_memory_peak_bytes = None
        if self._cuda_memory_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                self._gpu_memory_peak_bytes = 0
            except Exception:
                self._gpu_memory_peak_bytes = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if not self._started:
            return {
                "peak_cpu_ram_bytes": None,
                "cpu_utilization_percent": None,
                "peak_gpu_memory_bytes": None,
                "gpu_utilization_percent": None,
                "vram_allocation_over_time_bytes": None,
                "disk_offload_volume_bytes": None,
            }
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        avg_cpu = None
        if self._cpu_samples:
            avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples)

        gpu_peak = None
        if self._cuda_memory_available():
            try:
                gpu_peak = int(torch.cuda.max_memory_allocated())
            except Exception:
                gpu_peak = self._gpu_memory_peak_bytes

        return {
            "peak_cpu_ram_bytes": int(self._rss_peak_bytes),
            "cpu_utilization_percent": avg_cpu,
            "peak_gpu_memory_bytes": gpu_peak,
            "gpu_utilization_percent": None,
            "vram_allocation_over_time_bytes": None,
            "disk_offload_volume_bytes": None,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._rss_peak_bytes = max(self._rss_peak_bytes, self.process.memory_info().rss)
                self._cpu_samples.append(self.process.cpu_percent(interval=None))
                if self._cuda_memory_available():
                    current_gpu_memory = int(torch.cuda.memory_allocated())
                    peak_so_far = self._gpu_memory_peak_bytes or 0
                    self._gpu_memory_peak_bytes = max(peak_so_far, current_gpu_memory)
            except Exception:
                pass
            time.sleep(self.interval_seconds)

    def _cuda_memory_available(self) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False
