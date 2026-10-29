"""Composable dense retrieval with optional Sentence Transformers encoding."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import traceback
from typing import Any, Protocol, Sequence

import numpy as np

from drill_ir.data.models import Article, Query, RetrievalResult
from drill_ir.preprocessing import ArticleChunker, ArticleTruncationChunker, Chunk
from drill_ir.runtime_resources import cuda_snapshot, stage_heartbeat


@dataclass(frozen=True, slots=True)
class ChunkRetrievalResult:
    """Ranked retrieval evidence before article-level aggregation."""

    qid: int
    chunks: tuple[Chunk, ...]
    scores: tuple[float, ...]


class TextEncoder(Protocol):
    """Minimal encoder contract used by dense retrieval."""

    @property
    def cache_key(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SentenceTransformerEncoderConfig:
    model_name: str
    # Optional stable hub identifier used only for cache compatibility when
    # ``model_name`` is an equivalent local Kaggle snapshot path.
    cache_model_name: str | None = None
    revision: str | None = None
    batch_size: int = 32
    query_prefix: str = ""
    document_prefix: str = ""
    max_seq_length: int | None = None
    normalize_embeddings: bool = False
    trust_remote_code: bool = False
    device: str | list[str] = "auto"


def _isolated_sentence_transformer_encode(
    config: SentenceTransformerEncoderConfig,
    texts: list[str],
    devices: list[str],
    output_path: str,
    error_path: str,
) -> None:
    """Encode in a disposable owner process so native CUDA state cannot leak."""
    try:
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        print(
            f"[dense-worker] Loading model on CPU: {config.model_name}", flush=True
        )
        with stage_heartbeat("dense_checkpoint_cpu_materialization"):
            model = SentenceTransformer(
                config.model_name,
                revision=config.revision,
                trust_remote_code=config.trust_remote_code,
                device="cpu",
            )
        print(
            f"[dense-worker] Model materialized in "
            f"{time.perf_counter() - started:.1f}s; encoding "
            f"texts={len(texts)} devices={devices} batch_size={config.batch_size}",
            flush=True,
        )
        if config.max_seq_length is not None:
            model.max_seq_length = config.max_seq_length
        embeddings = model.encode(
            texts,
            batch_size=config.batch_size,
            device=devices,
            normalize_embeddings=config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        print(
            f"[dense-worker] Encoding complete in "
            f"{time.perf_counter() - started:.1f}s shape={list(embeddings.shape)}",
            flush=True,
        )
        np.save(output_path, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        raise


class SentenceTransformerEncoder:
    """Lazy Sentence Transformers adapter with query/document prefixes."""

    def __init__(self, config: SentenceTransformerEncoderConfig) -> None:
        if not config.model_name:
            raise ValueError("Dense encoder model_name cannot be empty")
        if config.batch_size <= 0:
            raise ValueError("Dense encoder batch_size must be positive")
        self.config = config
        self._model: Any = None
        self.last_device: str | list[str] | None = None

    @property
    def cache_key(self) -> str:
        payload = {
            "backend": "sentence_transformers",
            "model_name": self.config.cache_model_name or self.config.model_name,
            "revision": self.config.revision,
            "document_prefix": self.config.document_prefix,
            "max_seq_length": self.config.max_seq_length,
            "normalize_embeddings": self.config.normalize_embeddings,
            "trust_remote_code": self.config.trust_remote_code,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires the optional dense dependencies. "
                "Install with: python -m pip install -e '.[dense]'"
            ) from error
        started = time.perf_counter()
        print(f"[dense] Loading model: {self.config.model_name}", flush=True)
        with stage_heartbeat("dense_checkpoint_materialization"):
            self._model = SentenceTransformer(
                self.config.model_name,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
                # A list makes Sentence Transformers spawn one encoding worker per
                # device. Keep the source model in host memory so the parent does
                # not occupy cuda:0 alongside its worker's model replica.
                device=(
                    "cpu"
                    if isinstance(self.config.device, list)
                    else None if self.config.device == "auto" else self.config.device
                ),
            )
        if self.config.max_seq_length is not None:
            self._model.max_seq_length = self.config.max_seq_length
        print(
            f"[dense] Model ready in {time.perf_counter() - started:.1f}s: "
            f"{self.config.model_name}",
            flush=True,
        )
        return self._model

    def _resolve_device(self) -> str | list[str]:
        if self.config.device == "cpu":
            return self.config.device
        try:
            import torch
        except ImportError as error:
            if self.config.device != "auto":
                raise RuntimeError("A CUDA device was requested but PyTorch is unavailable") from error
            return "cpu"
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.config.device != "auto":
            requested = (
                self.config.device
                if isinstance(self.config.device, list)
                else [self.config.device]
            )
            invalid = [
                device
                for device in requested
                if not device.startswith("cuda:")
                or not device.removeprefix("cuda:").isdigit()
                or int(device.removeprefix("cuda:")) >= count
            ]
            if invalid:
                raise RuntimeError(
                    f"Requested CUDA devices {requested}, but only {count} device(s) are available"
                )
            return self.config.device
        if count == 0:
            return "cpu"
        if count == 1:
            return "cuda:0"
        return [f"cuda:{index}" for index in range(count)]

    def _encode(self, texts: Sequence[str], *, prefix: str) -> np.ndarray:
        trace = _CudaProcessTrace.from_environment(self.config.model_name)
        trace.start()
        try:
            prefixed = [prefix + text for text in texts]
            self.last_device = self._resolve_device()
            started = time.perf_counter()
            print(
                f"[dense] Encoding start model={self.config.model_name} "
                f"texts={len(prefixed)} batch_size={self.config.batch_size} "
                f"devices={self.last_device}",
                flush=True,
            )
            if isinstance(self.last_device, list):
                trace.record(
                    "before_isolated_encode",
                    resolved_device=self.last_device,
                    texts=len(prefixed),
                    batch_size=self.config.batch_size,
                )
                result = self._encode_multi_gpu_isolated(prefixed, self.last_device)
                trace.record("after_isolated_process_exit")
                print(
                    f"[dense] Encoding complete in "
                    f"{time.perf_counter() - started:.1f}s shape={list(result.shape)}",
                    flush=True,
                )
                return result
            trace.record("before_model_load")
            model = self._load_model()
            trace.record(
                "after_model_load",
                model_device=str(getattr(model, "device", "unknown")),
            )
            trace.record(
                "before_encode",
                resolved_device=self.last_device,
                texts=len(prefixed),
                batch_size=self.config.batch_size,
            )
            embeddings = model.encode(
                prefixed,
                batch_size=self.config.batch_size,
                device=self.last_device,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            trace.record("after_encode")
            result = np.asarray(embeddings, dtype=np.float32)
            print(
                f"[dense] Encoding complete in "
                f"{time.perf_counter() - started:.1f}s shape={list(result.shape)}",
                flush=True,
            )
            return result
        except BaseException as error:
            trace.record("encode_error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            trace.stop()

    def _encode_multi_gpu_isolated(
        self, texts: list[str], devices: list[str]
    ) -> np.ndarray:
        """Run SentenceTransformers' worker pool under a process we can join."""
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="drill-dense-encode-") as directory:
            output_path = str(Path(directory) / "embeddings.npy")
            error_path = str(Path(directory) / "error.txt")
            process = context.Process(
                target=_isolated_sentence_transformer_encode,
                args=(self.config, texts, devices, output_path, error_path),
                name=f"drill-encode-{Path(self.config.model_name).name}",
            )
            process.start()
            process.join()
            exit_code = process.exitcode
            process.close()
            if exit_code != 0 or not Path(output_path).is_file():
                detail = (
                    Path(error_path).read_text(encoding="utf-8")
                    if Path(error_path).is_file()
                    else "worker produced no error report"
                )
                raise RuntimeError(
                    f"Isolated multi-GPU encoding failed with exit code {exit_code}:\n{detail}"
                )
            return np.load(output_path, allow_pickle=False)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.config.document_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.config.query_prefix)

    def close(self) -> None:
        """Release model and CUDA allocations before a downstream model loads."""
        model, self._model = self._model, None
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class _CudaProcessTrace:
    """Opt-in nvidia-smi process sampler for diagnosing multi-process CUDA use."""

    def __init__(self, path: Path | None, interval_seconds: float, model_name: str) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.model_name = model_name
        self._stop = threading.Event()
        self._record_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_environment(cls, model_name: str) -> "_CudaProcessTrace":
        raw_path = os.environ.get("DRILL_CUDA_TRACE_PATH")
        raw_interval = os.environ.get("DRILL_CUDA_TRACE_INTERVAL_SECONDS", "0.5")
        try:
            interval = float(raw_interval)
        except ValueError:
            interval = 0.5
        return cls(Path(raw_path) if raw_path else None, max(interval, 0.1), model_name)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def start(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record("trace_start")
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self.record("trace_stop")

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.record("sample")

    def record(self, event: str, **details: Any) -> None:
        if not self.enabled:
            return
        with self._record_lock:
            self._record_unlocked(event, **details)

    def _record_unlocked(self, event: str, **details: Any) -> None:
        payload: dict[str, Any] = {
            "time_unix": time.time(),
            "event": event,
            "observer_pid": os.getpid(),
            "observer_ppid": os.getppid(),
            "model_name": self.model_name,
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
            rows = []
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3 or not parts[1].isdigit():
                    continue
                pid = int(parts[1])
                rows.append(
                    {
                        "gpu_uuid": parts[0],
                        "pid": pid,
                        "used_memory_mib": _optional_int(parts[2]),
                        **_process_identity(pid),
                    }
                )
            payload["compute_processes"] = rows
            if completed.returncode != 0:
                payload["nvidia_smi_error"] = completed.stderr.strip()
        except (OSError, subprocess.SubprocessError) as error:
            payload["nvidia_smi_error"] = f"{type(error).__name__}: {error}"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


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


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Dense embeddings contain a zero vector")
    return matrix / norms


class ExactVectorSearch:
    """Exact batched inner-product or cosine search with deterministic ties."""

    backend = "exact_numpy"

    def __init__(self, *, metric: str = "cosine", query_batch_size: int = 64) -> None:
        if metric not in {"cosine", "dot_product"}:
            raise ValueError("Dense search metric must be 'cosine' or 'dot_product'")
        if query_batch_size <= 0:
            raise ValueError("Dense search query_batch_size must be positive")
        self.metric = metric
        self.query_batch_size = query_batch_size

    def search(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        aids: Sequence[int],
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        documents = np.asarray(document_embeddings, dtype=np.float32)
        aid_array = np.asarray(aids, dtype=np.int64)
        if queries.ndim != 2 or documents.ndim != 2:
            raise ValueError("Dense embeddings must be two-dimensional matrices")
        if queries.shape[1] != documents.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match")
        if documents.shape[0] != len(aid_array):
            raise ValueError("Document embeddings and aid mapping have different lengths")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.metric == "cosine":
            queries = _l2_normalize(queries)
            documents = _l2_normalize(documents)

        result_k = min(top_k, documents.shape[0])
        all_indices: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        starts = range(0, queries.shape[0], self.query_batch_size)
        try:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc="Exact vector search", unit="batch")
        except ImportError:
            pass
        for start in starts:
            scores = queries[start : start + self.query_batch_size] @ documents.T
            for row in scores:
                # Full lexicographic ordering is the exact, deterministic
                # reference used to verify optimized search backends.
                selected = np.lexsort((aid_array, -row))[:result_k]
                all_indices.append(selected)
                all_scores.append(row[selected])
        return np.stack(all_indices), np.stack(all_scores)

    def search_grouped(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        aids: Sequence[int],
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate representation scores by aid with max before top-k."""
        queries = np.asarray(query_embeddings, dtype=np.float32)
        documents = np.asarray(document_embeddings, dtype=np.float32)
        aid_array = np.asarray(aids, dtype=np.int64)
        if queries.ndim != 2 or documents.ndim != 2:
            raise ValueError("Dense embeddings must be two-dimensional matrices")
        if queries.shape[1] != documents.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match")
        if documents.shape[0] != len(aid_array):
            raise ValueError("Document embeddings and aid mapping have different lengths")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.metric == "cosine":
            queries = _l2_normalize(queries)
            documents = _l2_normalize(documents)

        unique_aids, inverse = np.unique(aid_array, return_inverse=True)
        result_k = min(top_k, len(unique_aids))
        all_aids: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        starts = range(0, queries.shape[0], self.query_batch_size)
        try:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc="Grouped exact search", unit="batch")
        except ImportError:
            pass
        for start in starts:
            scores = queries[start : start + self.query_batch_size] @ documents.T
            for row in scores:
                grouped = np.full(len(unique_aids), -np.inf, dtype=np.float32)
                np.maximum.at(grouped, inverse, row)
                selected = np.lexsort((unique_aids, -grouped))[:result_k]
                all_aids.append(unique_aids[selected])
                all_scores.append(grouped[selected])
        return np.stack(all_aids), np.stack(all_scores)


class FaissFlatVectorSearch:
    """Exact FAISS flat search with deterministic aid tie-breaking.

    FAISS performs exhaustive inner-product search. Cosine search is implemented
    by L2-normalizing copied float32 vectors before they enter the index. Search
    depth expands when needed so article-level max aggregation remains exact and
    score ties at the requested boundary are available for deterministic sorting.
    """

    backend = "faiss_flat"

    def __init__(
        self,
        *,
        metric: str = "cosine",
        query_batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        if metric not in {"cosine", "dot_product"}:
            raise ValueError("FAISS search metric must be 'cosine' or 'dot_product'")
        if query_batch_size <= 0:
            raise ValueError("FAISS query_batch_size must be positive")
        if device != "auto" and device != "cpu" and not (
            device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("FAISS device must be 'auto', 'cpu', or 'cuda:N'")
        self.metric = metric
        self.query_batch_size = query_batch_size
        self.device = device
        self.resolved_device = "uninitialized"
        self._index: Any = None
        self._document_identity: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None
        self._gpu_resources: Any = None

    @staticmethod
    def _faiss() -> Any:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "FAISS search requires the dense dependency 'faiss-cpu'. "
                "Install with: python -m pip install -e '.[dense]'"
            ) from error
        return faiss

    def _resolve_device(self, faiss: Any) -> str:
        gpu_available = bool(
            hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > 0
        )
        if self.device == "auto":
            return "cuda:0" if gpu_available else "cpu"
        if self.device.startswith("cuda:") and not gpu_available:
            raise RuntimeError(
                f"Requested FAISS device {self.device}, but installed FAISS has no GPU support"
            )
        if self.device.startswith("cuda:"):
            gpu_id = int(self.device.removeprefix("cuda:"))
            if gpu_id >= faiss.get_num_gpus():
                raise RuntimeError(
                    f"Requested FAISS device {self.device}, but only {faiss.get_num_gpus()} GPU(s) exist"
                )
        return self.device

    def _prepare_index(self, document_embeddings: np.ndarray) -> Any:
        documents = np.asarray(document_embeddings, dtype=np.float32)
        if documents.ndim != 2:
            raise ValueError("Dense embeddings must be two-dimensional matrices")
        identity = (id(document_embeddings), documents.shape, documents.strides)
        if self._index is not None and identity == self._document_identity:
            return self._index
        self.close()
        faiss = self._faiss()
        vectors = np.ascontiguousarray(documents, dtype=np.float32)
        if self.metric == "cosine":
            vectors = vectors.copy()
            faiss.normalize_L2(vectors)
            if np.any(np.linalg.norm(vectors, axis=1) == 0):
                raise ValueError("Dense embeddings contain a zero vector")
        cpu_index = faiss.IndexFlatIP(vectors.shape[1])
        cpu_index.add(vectors)
        self.resolved_device = self._resolve_device(faiss)
        if self.resolved_device.startswith("cuda:"):
            gpu_id = int(self.resolved_device.removeprefix("cuda:"))
            self._gpu_resources = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(
                self._gpu_resources, gpu_id, cpu_index
            )
        else:
            self._index = cpu_index
        self._document_identity = identity
        return self._index

    def _prepare_queries(self, query_embeddings: np.ndarray) -> np.ndarray:
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("Dense embeddings must be two-dimensional matrices")
        if self.metric == "cosine":
            queries = queries.copy()
            self._faiss().normalize_L2(queries)
            if np.any(np.linalg.norm(queries, axis=1) == 0):
                raise ValueError("Dense embeddings contain a zero vector")
        return queries

    @staticmethod
    def _standard_depth_is_safe(
        scores: np.ndarray, *, requested_k: int, searched_k: int, total: int
    ) -> bool:
        if searched_k >= total or requested_k >= total:
            return True
        return bool(scores[searched_k - 1] < scores[requested_k - 1])

    @staticmethod
    def _grouped_depth_is_safe(
        scores: np.ndarray,
        indices: np.ndarray,
        aids: np.ndarray,
        *,
        requested_k: int,
        searched_k: int,
        total: int,
    ) -> bool:
        best: dict[int, float] = {}
        for score, position in zip(scores, indices):
            aid = int(aids[int(position)])
            best[aid] = max(best.get(aid, -float("inf")), float(score))
        if len(best) < requested_k:
            return searched_k >= total
        threshold = sorted(best.items(), key=lambda item: (-item[1], item[0]))[
            requested_k - 1
        ][1]
        return searched_k >= total or bool(scores[searched_k - 1] < threshold)

    def _search_batch(
        self,
        index: Any,
        queries: np.ndarray,
        *,
        requested_k: int,
        aids: np.ndarray,
        grouped: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = int(index.ntotal)
        searched_k = min(total, max(requested_k + 64, requested_k * 2))
        while True:
            scores, indices = index.search(queries, searched_k)
            safe = []
            for row in range(len(queries)):
                if grouped:
                    safe.append(
                        self._grouped_depth_is_safe(
                            scores[row], indices[row], aids,
                            requested_k=requested_k,
                            searched_k=searched_k,
                            total=total,
                        )
                    )
                else:
                    safe.append(
                        self._standard_depth_is_safe(
                            scores[row], requested_k=requested_k,
                            searched_k=searched_k,
                            total=total,
                        )
                    )
            if all(safe) or searched_k >= total:
                return scores, indices
            searched_k = min(total, searched_k * 2)

    def search(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        aids: Sequence[int],
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        documents = np.asarray(document_embeddings, dtype=np.float32)
        aid_array = np.asarray(aids, dtype=np.int64)
        if documents.shape[0] != len(aid_array):
            raise ValueError("Document embeddings and aid mapping have different lengths")
        if top_k <= 0 or documents.shape[0] == 0:
            raise ValueError("top_k must be positive and document embeddings non-empty")
        queries = self._prepare_queries(query_embeddings)
        if queries.shape[1] != documents.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match")
        index = self._prepare_index(documents)
        result_k = min(top_k, len(aid_array))
        all_indices: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for start in range(0, len(queries), self.query_batch_size):
            scores, indices = self._search_batch(
                index, queries[start : start + self.query_batch_size],
                requested_k=result_k, aids=aid_array, grouped=False,
            )
            for row_scores, row_indices in zip(scores, indices):
                order = np.lexsort((aid_array[row_indices], -row_scores))[:result_k]
                all_indices.append(row_indices[order])
                all_scores.append(row_scores[order])
        return np.stack(all_indices), np.stack(all_scores)

    def search_grouped(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        aids: Sequence[int],
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        documents = np.asarray(document_embeddings, dtype=np.float32)
        aid_array = np.asarray(aids, dtype=np.int64)
        if documents.shape[0] != len(aid_array):
            raise ValueError("Document embeddings and aid mapping have different lengths")
        if top_k <= 0 or documents.shape[0] == 0:
            raise ValueError("top_k must be positive and document embeddings non-empty")
        queries = self._prepare_queries(query_embeddings)
        if queries.shape[1] != documents.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match")
        index = self._prepare_index(documents)
        result_k = min(top_k, len(np.unique(aid_array)))
        all_aids: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for start in range(0, len(queries), self.query_batch_size):
            scores, indices = self._search_batch(
                index, queries[start : start + self.query_batch_size],
                requested_k=result_k, aids=aid_array, grouped=True,
            )
            for row_scores, row_indices in zip(scores, indices):
                best: dict[int, float] = {}
                for score, position in zip(row_scores, row_indices):
                    aid = int(aid_array[int(position)])
                    best[aid] = max(best.get(aid, -float("inf")), float(score))
                selected = sorted(best.items(), key=lambda item: (-item[1], item[0]))[
                    :result_k
                ]
                all_aids.append(np.asarray([aid for aid, _ in selected], dtype=np.int64))
                all_scores.append(np.asarray([score for _, score in selected], dtype=np.float32))
        return np.stack(all_aids), np.stack(all_scores)

    def close(self) -> None:
        self._index = None
        self._document_identity = None
        self._gpu_resources = None
        self.resolved_device = "uninitialized"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class EmbeddingCache:
    """Validated on-disk cache for corpus embeddings and their aid mapping."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def corpus_fingerprint(articles: Sequence[Article]) -> str:
        digest = hashlib.sha256()
        for article in articles:
            digest.update(str(article.aid).encode())
            digest.update(b"\0")
            digest.update(article.law_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(article.text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _directory(self, corpus_fingerprint: str, encoder_key: str) -> Path:
        key = hashlib.sha256(f"{corpus_fingerprint}\0{encoder_key}".encode()).hexdigest()[:20]
        return self.root / key

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load(
        self, *, corpus_fingerprint: str, encoder_key: str, aids: Sequence[int]
    ) -> np.ndarray | None:
        directory = self._directory(corpus_fingerprint, encoder_key)
        metadata_path = directory / "metadata.json"
        embeddings_path = directory / "embeddings.npy"
        if not metadata_path.is_file() or not embeddings_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_aids = list(aids)
        if (
            metadata.get("corpus_fingerprint") != corpus_fingerprint
            or metadata.get("encoder_key") != encoder_key
            or metadata.get("aids") != expected_aids
        ):
            return None
        expected_sha256 = metadata.get("embeddings_sha256")
        if expected_sha256 is not None and expected_sha256 != self._file_sha256(embeddings_path):
            return None
        # Keep the immutable matrix file-backed until a search backend truly
        # needs resident/normalized storage. This avoids an eager duplicate in
        # multi-dense processes and preserves the exact float32 values.
        embeddings = np.load(embeddings_path, allow_pickle=False, mmap_mode="r")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_aids):
            raise ValueError(f"Invalid dense embedding cache: {directory}")
        return np.asarray(embeddings, dtype=np.float32)

    def save(
        self,
        embeddings: np.ndarray,
        *,
        corpus_fingerprint: str,
        encoder_key: str,
        aids: Sequence[int],
    ) -> Path:
        directory = self._directory(corpus_fingerprint, encoder_key)
        directory.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(embeddings, dtype=np.float32)
        np.save(directory / "embeddings.npy", matrix, allow_pickle=False)
        metadata = {
            "corpus_fingerprint": corpus_fingerprint,
            "encoder_key": encoder_key,
            "aids": list(aids),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "embeddings_sha256": self._file_sha256(directory / "embeddings.npy"),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory


class DenseChunkCache:
    """Portable, checksummed cache of deterministic dense chunk records."""

    FORMAT_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, corpus_fingerprint: str, chunker_key: str) -> Path:
        identity = json.dumps(
            {
                "format_version": self.FORMAT_VERSION,
                "corpus_fingerprint": corpus_fingerprint,
                "chunker_key": chunker_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self.root / hashlib.sha256(identity.encode()).hexdigest()[:20]

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load(
        self, *, corpus_fingerprint: str, chunker_key: str
    ) -> tuple[list[Chunk], dict[str, Any]] | None:
        directory = self._directory(corpus_fingerprint, chunker_key)
        metadata_path = directory / "metadata.json"
        chunks_path = directory / "chunks.jsonl"
        if not metadata_path.is_file() or not chunks_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("format_version") != self.FORMAT_VERSION
            or metadata.get("corpus_fingerprint") != corpus_fingerprint
            or metadata.get("chunker_key") != chunker_key
            or metadata.get("chunks_sha256") != self._file_sha256(chunks_path)
        ):
            return None
        chunks: list[Chunk] = []
        with chunks_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=str(record["chunk_id"]),
                        aid=int(record["aid"]),
                        law_id=str(record["law_id"]),
                        chunk_index=int(record["chunk_index"]),
                        chunk_type=str(record["chunk_type"]),
                        text=str(record["text"]),
                        char_start=int(record["char_start"]),
                        char_end=int(record["char_end"]),
                        token_count=(
                            int(record["token_count"])
                            if record.get("token_count") is not None
                            else None
                        ),
                        parse_status=str(record.get("parse_status", "not_applicable")),
                        structure_path=tuple(str(value) for value in record.get("structure_path", ())),
                        source_spans=tuple(
                            (int(span[0]), int(span[1]))
                            for span in record.get("source_spans", ())
                        ),
                    )
                )
        if len(chunks) != int(metadata.get("chunk_count", -1)):
            raise ValueError(f"Invalid dense chunk cache: {directory}")
        return chunks, dict(metadata.get("chunking_stats", {}))

    def save(
        self,
        chunks: Sequence[Chunk],
        *,
        corpus_fingerprint: str,
        chunker_key: str,
        chunking_stats: dict[str, Any] | None = None,
    ) -> Path:
        directory = self._directory(corpus_fingerprint, chunker_key)
        directory.mkdir(parents=True, exist_ok=True)
        chunks_path = directory / "chunks.jsonl"
        temporary_path = directory / "chunks.jsonl.tmp"
        with temporary_path.open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "aid": chunk.aid,
                    "law_id": chunk.law_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                    "parse_status": chunk.parse_status,
                    "structure_path": list(chunk.structure_path),
                    "source_spans": [list(span) for span in chunk.source_spans],
                }
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        os.replace(temporary_path, chunks_path)
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "corpus_fingerprint": corpus_fingerprint,
            "chunker_key": chunker_key,
            "chunk_count": len(chunks),
            "chunks_sha256": self._file_sha256(chunks_path),
            "chunking_stats": chunking_stats or {},
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory


class DenseRetriever:
    """Bi-encoder retriever composed from encoder, cache, and search components."""

    def __init__(
        self,
        *,
        encoder: TextEncoder,
        search: ExactVectorSearch | FaissFlatVectorSearch,
        cache: EmbeddingCache | None = None,
        chunk_cache: DenseChunkCache | None = None,
        chunker: ArticleChunker | None = None,
        rocchio_enabled: bool = False,
        rocchio_feedback_documents: int = 5,
        rocchio_alpha: float = 1.0,
        rocchio_beta: float = 0.25,
        rocchio_gamma: float = 0.0,
        rocchio_negative_documents: int = 0,
        rocchio_candidate_depth: int = 20,
        require_cache_hit: bool = False,
    ) -> None:
        if rocchio_feedback_documents <= 0 or rocchio_candidate_depth <= 0:
            raise ValueError("Rocchio feedback documents and candidate depth must be positive")
        if rocchio_alpha < 0 or rocchio_beta < 0 or rocchio_gamma < 0:
            raise ValueError("Rocchio alpha, beta, and gamma must be non-negative")
        if rocchio_alpha == 0 and rocchio_beta == 0:
            raise ValueError("Rocchio requires alpha or beta to be positive")
        if rocchio_negative_documents < 0:
            raise ValueError("Rocchio negative_documents cannot be negative")
        if rocchio_negative_documents and rocchio_gamma == 0:
            raise ValueError("Rocchio negative_documents requires gamma > 0")
        if rocchio_gamma > 0 and rocchio_negative_documents == 0:
            raise ValueError("Rocchio gamma > 0 requires negative_documents")
        self.encoder = encoder
        self.search = search
        self.cache = cache
        self.chunk_cache = chunk_cache
        self.chunker = chunker or ArticleTruncationChunker()
        self._aids: list[int] = []
        self._chunks: list[Chunk] = []
        self._document_embeddings: np.ndarray | None = None
        self.cache_hit = False
        self.chunk_cache_hit = False
        self.raw_corpus_fingerprint: str | None = None
        self.corpus_fingerprint: str | None = None
        self.rocchio_enabled = rocchio_enabled
        self.rocchio_feedback_documents = rocchio_feedback_documents
        self.rocchio_alpha = rocchio_alpha
        self.rocchio_beta = rocchio_beta
        self.rocchio_gamma = rocchio_gamma
        self.rocchio_negative_documents = rocchio_negative_documents
        self.rocchio_candidate_depth = rocchio_candidate_depth
        self.require_cache_hit = require_cache_hit
        if self.require_cache_hit and self.cache is None:
            raise ValueError("require_cache_hit requires an enabled dense cache")
        self._last_feedback: list[dict[str, object]] = []
        self._cached_query_key: tuple[tuple[int, str], ...] | None = None
        self._cached_query_embeddings: np.ndarray | None = None

    def prepare_chunks(
        self, articles: Sequence[Article], *, require_existing: bool = False
    ) -> list[Chunk]:
        """Load or create deterministic chunks without encoding documents."""
        if not articles:
            raise ValueError("Cannot fit dense retrieval on an empty corpus")
        started = time.perf_counter()
        print(
            f"[dense] Fingerprinting raw corpus rows={len(articles):,}", flush=True
        )
        self.raw_corpus_fingerprint = EmbeddingCache.corpus_fingerprint(articles)
        print(
            f"[dense] Raw corpus fingerprint ready in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        cached_chunks = None
        if self.chunk_cache is not None:
            cached_chunks = self.chunk_cache.load(
                corpus_fingerprint=self.raw_corpus_fingerprint,
                chunker_key=self.chunker.cache_key,
            )
        if cached_chunks is not None:
            self._chunks, chunking_stats = cached_chunks
            self.chunk_cache_hit = True
            if hasattr(self.chunker, "last_stats"):
                self.chunker.last_stats = chunking_stats
            print("[dense] Chunk cache hit", flush=True)
        else:
            self.chunk_cache_hit = False
            if require_existing:
                raise RuntimeError("Required dense chunk cache is missing or invalid")
            print(
                f"[dense] Chunking {len(articles):,} articles with "
                f"{type(self.chunker).__name__}",
                flush=True,
            )
            self._chunks = self.chunker.chunk_articles(articles)
            if self.chunk_cache is not None:
                self.chunk_cache.save(
                    self._chunks,
                    corpus_fingerprint=self.raw_corpus_fingerprint,
                    chunker_key=self.chunker.cache_key,
                    chunking_stats=getattr(self.chunker, "last_stats", {}),
                )
                print("[dense] Saved chunk cache", flush=True)
        if not self._chunks:
            raise ValueError("Chunker produced no dense representations")
        print(f"[dense] Prepared {len(self._chunks):,} representations", flush=True)
        return self._chunks

    def fit(self, articles: Sequence[Article]) -> None:
        if not articles:
            raise ValueError("Cannot fit dense retrieval on an empty corpus")
        self._cached_query_key = None
        self._cached_query_embeddings = None
        self._last_feedback = []
        self.prepare_chunks(articles)
        self._aids = [chunk.aid for chunk in self._chunks]
        representations = [
            Article(aid=chunk.aid, law_id=chunk.law_id, text=chunk.text)
            for chunk in self._chunks
        ]
        started = time.perf_counter()
        print(
            f"[dense] Fingerprinting representations rows={len(representations):,}",
            flush=True,
        )
        fingerprint = EmbeddingCache.corpus_fingerprint(representations)
        print(
            f"[dense] Representation fingerprint ready in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.corpus_fingerprint = fingerprint
        representation_key = json.dumps(
            {"encoder": self.encoder.cache_key, "chunker": self.chunker.cache_key},
            sort_keys=True,
            separators=(",", ":"),
        )
        diagnostic_model_name = str(
            getattr(getattr(self.encoder, "config", None), "model_name", type(self.encoder).__name__)
        )
        embeddings = None
        if self.cache is not None:
            cuda_snapshot(
                "dense_cache:before_load", model_name=diagnostic_model_name
            )
            embeddings = self.cache.load(
                corpus_fingerprint=fingerprint,
                encoder_key=representation_key,
                aids=self._aids,
            )
            self.cache_hit = embeddings is not None
            cuda_snapshot(
                "dense_cache:after_load",
                model_name=diagnostic_model_name,
                cache_hit=self.cache_hit,
            )
            print(
                f"[dense] Embedding cache {'hit' if self.cache_hit else 'miss'}",
                flush=True,
            )
        if embeddings is None:
            if self.require_cache_hit:
                raise RuntimeError("Required dense embedding cache is missing or invalid")
            print("[dense] Encoding corpus representations", flush=True)
            embeddings = self.encoder.encode_documents([chunk.text for chunk in self._chunks])
            if embeddings.ndim != 2 or embeddings.shape[0] != len(self._chunks):
                raise ValueError("Encoder returned an invalid document embedding matrix")
            if self.cache is not None:
                self.cache.save(
                    embeddings,
                    corpus_fingerprint=fingerprint,
                    encoder_key=representation_key,
                    aids=self._aids,
                )
                print("[dense] Saved corpus embedding cache", flush=True)
        self._document_embeddings = embeddings
        cuda_snapshot(
            "dense_fit:ready",
            model_name=diagnostic_model_name,
            embedding_shape=list(embeddings.shape),
        )
        print(f"[dense] Index ready with shape {list(embeddings.shape)}", flush=True)

    def run_metadata(self) -> dict[str, Any]:
        shape = list(self._document_embeddings.shape) if self._document_embeddings is not None else None
        device = getattr(self.encoder, "last_device", None)
        return {
            "encoder_cache_key": self.encoder.cache_key,
            "raw_corpus_fingerprint": self.raw_corpus_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "embedding_shape": shape,
            "embedding_cache_hit": self.cache_hit,
            "embedding_cache_required": self.require_cache_hit,
            "chunk_cache_enabled": self.chunk_cache is not None,
            "chunk_cache_hit": self.chunk_cache_hit,
            "search_backend": self.search.backend,
            "search_metric": self.search.metric,
            "search_device": getattr(self.search, "resolved_device", "cpu"),
            "encoder_device": device,
            "chunker_cache_key": self.chunker.cache_key,
            "chunking": getattr(self.chunker, "last_stats", {"method": "article_truncate"}),
            "representations": len(self._chunks),
            "unique_aids": len(set(self._aids)),
            "aggregation": "max",
            "feedback": {
                "enabled": self.rocchio_enabled,
                "method": "rocchio" if self.rocchio_enabled else "none",
                "feedback_documents": self.rocchio_feedback_documents,
                "alpha": self.rocchio_alpha,
                "beta": self.rocchio_beta,
                "gamma": self.rocchio_gamma,
                "negative_documents": self.rocchio_negative_documents,
                "candidate_depth": self.rocchio_candidate_depth,
                "unique_aids": True,
            },
        }

    def close(self) -> None:
        """Drop corpus/query matrices, chunks, and encoder resources."""
        close_search = getattr(self.search, "close", None)
        if callable(close_search):
            close_search()
        close_encoder = getattr(self.encoder, "close", None)
        if callable(close_encoder):
            close_encoder()
        self._document_embeddings = None
        self._cached_query_embeddings = None
        self._cached_query_key = None
        self._chunks = []
        self._aids = []
        gc.collect()

    def _feedback_embeddings(self, queries: Sequence[Query]) -> np.ndarray:
        if self._document_embeddings is None:
            raise RuntimeError("fit() must be called before retrieval")
        key = tuple((query.qid, query.question) for query in queries)
        if key == self._cached_query_key and self._cached_query_embeddings is not None:
            return self._cached_query_embeddings
        original = self.encoder.encode_queries([query.question for query in queries])
        if not self.rocchio_enabled:
            adapted = original
            self._last_feedback = []
        else:
            rocchio_queries = (
                _l2_normalize(original) if self.search.metric == "cosine" else original
            )
            depth = max(
                self.rocchio_candidate_depth,
                self.rocchio_feedback_documents + self.rocchio_negative_documents,
            )
            indices, _ = self.search.search(
                original, self._document_embeddings, self._aids, top_k=depth
            )
            adapted_rows = []
            records = []
            for row, query in enumerate(queries):
                positive_indices: list[int] = []
                seen_aids: set[int] = set()
                for index in indices[row]:
                    position = int(index)
                    aid = self._aids[position]
                    if aid in seen_aids:
                        continue
                    seen_aids.add(aid)
                    positive_indices.append(position)
                    if len(positive_indices) == self.rocchio_feedback_documents:
                        break
                negative_indices: list[int] = []
                if self.rocchio_negative_documents:
                    negative_seen: set[int] = set()
                    for index in reversed(indices[row]):
                        position = int(index)
                        aid = self._aids[position]
                        if aid in seen_aids or aid in negative_seen:
                            continue
                        negative_seen.add(aid)
                        negative_indices.append(position)
                        if len(negative_indices) == self.rocchio_negative_documents:
                            break
                positive_vectors = _l2_normalize(
                    self._document_embeddings[positive_indices]
                )
                vector = self.rocchio_alpha * rocchio_queries[row] + self.rocchio_beta * positive_vectors.mean(axis=0)
                if negative_indices:
                    negative_vectors = _l2_normalize(
                        self._document_embeddings[negative_indices]
                    )
                    vector = vector - self.rocchio_gamma * negative_vectors.mean(axis=0)
                if not np.any(vector):
                    raise ValueError(f"Rocchio produced a zero vector for qid={query.qid}")
                adapted_rows.append(vector)
                records.append(
                    {
                        "qid": query.qid,
                        "feedback_aids": [self._aids[index] for index in positive_indices],
                        "negative_aids": [self._aids[index] for index in negative_indices],
                    }
                )
            adapted = np.asarray(adapted_rows, dtype=np.float32)
            self._last_feedback = records
        self._cached_query_key = key
        self._cached_query_embeddings = np.asarray(adapted, dtype=np.float32)
        return self._cached_query_embeddings

    def feedback_records(self) -> list[dict[str, object]]:
        return list(self._last_feedback)

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if self._document_embeddings is None:
            raise RuntimeError("fit() must be called before retrieve()")
        if not queries:
            return []
        query_embeddings = self._feedback_embeddings(queries)
        aids, scores = self.search.search_grouped(
            query_embeddings,
            self._document_embeddings,
            self._aids,
            top_k=top_k,
        )
        return [
            RetrievalResult(
                qid=query.qid,
                aids=tuple(int(aid) for aid in aids[row]),
                scores=tuple(float(score) for score in scores[row]),
            )
            for row, query in enumerate(queries)
        ]

    def retrieve_chunks(
        self, queries: Sequence[Query], top_k: int
    ) -> list[ChunkRetrievalResult]:
        """Return ranked chunk evidence without aggregating duplicate article IDs."""
        if self._document_embeddings is None:
            raise RuntimeError("fit() must be called before retrieve_chunks()")
        if not queries:
            return []
        query_embeddings = self._feedback_embeddings(queries)
        indices, scores = self.search.search(
            query_embeddings,
            self._document_embeddings,
            self._aids,
            top_k=top_k,
        )
        return [
            ChunkRetrievalResult(
                qid=query.qid,
                chunks=tuple(self._chunks[int(index)] for index in indices[row]),
                scores=tuple(float(score) for score in scores[row]),
            )
            for row, query in enumerate(queries)
        ]
