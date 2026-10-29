"""CUDA lifecycle diagnostics and deterministic stage-boundary cleanup."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any


@contextmanager
def stage_heartbeat(label: str, *, interval_seconds: float = 30.0):
    """Emit elapsed-time heartbeats while a blocking native/library call runs."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    started = time.perf_counter()
    stopped = threading.Event()
    print(
        f"[stage-heartbeat] start stage={label} pid={os.getpid()}",
        flush=True,
    )

    def report() -> None:
        while not stopped.wait(interval_seconds):
            print(
                f"[stage-heartbeat] alive stage={label} pid={os.getpid()} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    reporter = threading.Thread(
        target=report,
        name=f"drill-heartbeat-{label}",
        daemon=True,
    )
    reporter.start()
    status = "failed"
    try:
        yield
        status = "completed"
    finally:
        stopped.set()
        reporter.join(timeout=1.0)
        print(
            f"[stage-heartbeat] stop stage={label} pid={os.getpid()} "
            f"status={status} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )


def _process_identity(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"ppid": None, "command": None}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                result["ppid"] = int(line.split()[1])
                break
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        result["command"] = command.decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        pass
    return result


def cuda_snapshot(label: str, **details: Any) -> dict[str, Any]:
    """Measure global per-PID VRAM and this process's PyTorch allocator state."""
    payload: dict[str, Any] = {
        "time_unix": time.time(),
        "label": label,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        **details,
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        processes = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            process_pid = int(parts[1])
            try:
                used_memory = int(parts[2])
            except ValueError:
                used_memory = None
            processes.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": process_pid,
                    "used_memory_mib": used_memory,
                    **_process_identity(process_pid),
                }
            )
        payload["compute_processes"] = processes
        if completed.returncode != 0:
            payload["nvidia_smi_error"] = completed.stderr.strip()
    except (OSError, subprocess.SubprocessError) as error:
        payload["nvidia_smi_error"] = f"{type(error).__name__}: {error}"

    # Do not import torch merely to observe process start: importing/initializing
    # CUDA would itself perturb the lifecycle being measured.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        allocator = []
        for index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            allocator.append(
                {
                    "device": f"cuda:{index}",
                    "free_mib": free_bytes / 2**20,
                    "total_mib": total_bytes / 2**20,
                    "allocated_mib": torch.cuda.memory_allocated(index) / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved(index) / 2**20,
                }
            )
        payload["torch_cuda"] = allocator
    else:
        payload["torch_cuda"] = []

    path_value = os.environ.get("DRILL_CUDA_LIFECYCLE_PATH")
    if not path_value and (dense_trace := os.environ.get("DRILL_CUDA_TRACE_PATH")):
        dense_path = Path(dense_trace)
        path_value = str(dense_path.with_name("cuda_lifecycle_trace.jsonl"))
    if path_value:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    own = [
        row for row in payload.get("compute_processes", []) if row["pid"] == os.getpid()
    ]
    all_processes = [
        (row["gpu_uuid"], row["pid"], row["used_memory_mib"], row["ppid"])
        for row in payload.get("compute_processes", [])
    ]
    print(
        f"[cuda-lifecycle] {label} pid={os.getpid()} "
        f"process_vram={[(row['gpu_uuid'], row['used_memory_mib']) for row in own]} "
        f"all_processes={all_processes} "
        f"torch={payload['torch_cuda']}",
        flush=True,
    )
    return payload


def release_stage_resources(owner: Any, *, stage: str) -> None:
    """Close a stage owner and reclaim all releasable PyTorch/CUDA IPC state."""
    cuda_snapshot(f"{stage}:before_close")
    close = getattr(owner, "close", None)
    if callable(close):
        close()
    close = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            gc.collect()
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
    except ImportError:
        pass
    cuda_snapshot(f"{stage}:after_close")
