"""Composable dense retrieval with optional Sentence Transformers encoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from drill_ir.data.models import Article, Query, RetrievalResult


class TextEncoder(Protocol):
    """Minimal encoder contract used by dense retrieval."""

    @property
    def cache_key(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SentenceTransformerEncoderConfig:
    model_name: str
    revision: str | None = None
    batch_size: int = 32
    query_prefix: str = ""
    document_prefix: str = ""
    max_seq_length: int | None = None
    normalize_embeddings: bool = False
    trust_remote_code: bool = False
    device: str | list[str] = "auto"


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
            "model_name": self.config.model_name,
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
        self._model = SentenceTransformer(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.max_seq_length is not None:
            self._model.max_seq_length = self.config.max_seq_length
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
        model = self._load_model()
        prefixed = [prefix + text for text in texts]
        self.last_device = self._resolve_device()
        embeddings = model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            device=self.last_device,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.config.document_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.config.query_prefix)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Dense embeddings contain a zero vector")
    return matrix / norms


class ExactVectorSearch:
    """Exact batched inner-product or cosine search with deterministic ties."""

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
        for start in range(0, queries.shape[0], self.query_batch_size):
            scores = queries[start : start + self.query_batch_size] @ documents.T
            for row in scores:
                # Full lexicographic ordering is the exact, deterministic
                # baseline. FAISS/ANN can be introduced as a separate backend.
                selected = np.lexsort((aid_array, -row))[:result_k]
                all_indices.append(selected)
                all_scores.append(row[selected])
        return np.stack(all_indices), np.stack(all_scores)


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
        embeddings = np.load(embeddings_path, allow_pickle=False)
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
        search: ExactVectorSearch,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.encoder = encoder
        self.search = search
        self.cache = cache
        self._aids: list[int] = []
        self._document_embeddings: np.ndarray | None = None
        self.cache_hit = False
        self.corpus_fingerprint: str | None = None

    def fit(self, articles: Sequence[Article]) -> None:
        if not articles:
            raise ValueError("Cannot fit dense retrieval on an empty corpus")
        self._aids = [article.aid for article in articles]
        fingerprint = EmbeddingCache.corpus_fingerprint(articles)
        self.corpus_fingerprint = fingerprint
        embeddings = None
        if self.cache is not None:
            embeddings = self.cache.load(
                corpus_fingerprint=fingerprint,
                encoder_key=self.encoder.cache_key,
                aids=self._aids,
            )
            self.cache_hit = embeddings is not None
        if embeddings is None:
            embeddings = self.encoder.encode_documents([article.text for article in articles])
            if embeddings.ndim != 2 or embeddings.shape[0] != len(articles):
                raise ValueError("Encoder returned an invalid document embedding matrix")
            if self.cache is not None:
                self.cache.save(
                    embeddings,
                    corpus_fingerprint=fingerprint,
                    encoder_key=self.encoder.cache_key,
                    aids=self._aids,
                )
        self._document_embeddings = embeddings

    def run_metadata(self) -> dict[str, Any]:
        shape = list(self._document_embeddings.shape) if self._document_embeddings is not None else None
        device = getattr(self.encoder, "last_device", None)
        return {
            "encoder_cache_key": self.encoder.cache_key,
            "corpus_fingerprint": self.corpus_fingerprint,
            "embedding_shape": shape,
            "embedding_cache_hit": self.cache_hit,
            "search_backend": "exact",
            "search_metric": self.search.metric,
            "encoder_device": device,
        }

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if self._document_embeddings is None:
            raise RuntimeError("fit() must be called before retrieve()")
        if not queries:
            return []
        query_embeddings = self.encoder.encode_queries([query.question for query in queries])
        indices, scores = self.search.search(
            query_embeddings,
            self._document_embeddings,
            self._aids,
            top_k=top_k,
        )
        return [
            RetrievalResult(
                qid=query.qid,
                aids=tuple(self._aids[index] for index in indices[row]),
                scores=tuple(float(score) for score in scores[row]),
            )
            for row, query in enumerate(queries)
        ]
