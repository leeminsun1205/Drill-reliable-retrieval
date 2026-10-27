"""Lightweight process resource telemetry for reproducible experiments."""

from __future__ import annotations

import resource
import sys
from typing import Any


def reset_cuda_peaks() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
    except ImportError:
        pass


def resource_peaks() -> dict[str, Any]:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mib = rss / (1024.0 if sys.platform != "darwin" else 1024.0**2)
    cuda: list[dict[str, Any]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                cuda.append({
                    "device": device,
                    "name": torch.cuda.get_device_name(device),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
                })
    except ImportError:
        pass
    return {"peak_rss_mib": peak_rss_mib, "cuda": cuda}
