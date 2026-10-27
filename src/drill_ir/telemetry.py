"""Lightweight process resource telemetry for reproducible experiments."""

from __future__ import annotations

import resource
import sys
from typing import Any


def reset_cuda_peaks() -> None:
    """Reset supported CUDA counters without making telemetry a run dependency."""
    try:
        import torch
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(device):
                        torch.cuda.reset_peak_memory_stats()
                except (RuntimeError, ValueError):
                    # Some CUDA/PyTorch builds expose a visible device count but
                    # reject peak-counter operations for a process-local device.
                    continue
    except (ImportError, RuntimeError):
        pass


def resource_peaks() -> dict[str, Any]:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mib = rss / (1024.0 if sys.platform != "darwin" else 1024.0**2)
    cuda: list[dict[str, Any]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(device):
                        cuda.append({
                            "device": device,
                            "name": torch.cuda.get_device_name(),
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                        })
                except (RuntimeError, ValueError):
                    cuda.append({"device": device, "available": False})
    except (ImportError, RuntimeError):
        pass
    return {"peak_rss_mib": peak_rss_mib, "cuda": cuda}
