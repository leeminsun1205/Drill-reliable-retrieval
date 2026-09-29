"""Build retrieval components from explicit experiment configuration."""

from pathlib import Path
from typing import Any

from .base import Retriever
from .bm25 import BM25Retriever
from .dense import (
    DenseRetriever,
    EmbeddingCache,
    ExactVectorSearch,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
)
from drill_ir.preprocessing import build_token_processor


def _dense_device(runtime: dict[str, Any]) -> str | list[str]:
    accelerator = runtime.get("accelerator", "auto")
    devices = runtime.get("devices", "auto")
    if accelerator == "cpu":
        return "cpu"
    if devices == "auto":
        return "auto"
    if isinstance(devices, bool) or not isinstance(devices, int) or devices <= 0:
        raise ValueError("runtime.devices must be 'auto' or a positive integer")
    if accelerator not in {"auto", "cuda"}:
        raise ValueError("runtime.accelerator must be 'auto', 'cpu', or 'cuda'")
    return [f"cuda:{index}" for index in range(devices)] if devices > 1 else "cuda:0"


def build_retriever(
    method: str,
    retrieval_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> Retriever:
    """Build one first-stage retriever behind the common Retriever protocol."""
    runtime = runtime_config or {}
    if method == "bm25":
        params = retrieval_config.get("bm25", {})
        return BM25Retriever(
            k1=float(params.get("k1", 1.2)),
            b=float(params.get("b", 0.75)),
            text_processor=build_token_processor(params.get("preprocessing")),
            min_document_frequency=int(params.get("min_document_frequency", 1)),
            max_document_frequency=(
                int(params["max_document_frequency"])
                if params.get("max_document_frequency") is not None
                else None
            ),
            max_document_frequency_ratio=float(
                params.get("max_document_frequency_ratio", 1.0)
            ),
        )
    if method != "dense":
        raise ValueError(f"No retriever factory registered for method={method!r}")

    dense = retrieval_config.get("dense")
    if not isinstance(dense, dict):
        raise ValueError("retrieval.dense must be a YAML mapping")
    encoder_config = dense.get("encoder")
    search_config = dense.get("search")
    cache_config = dense.get("cache", {})
    if not isinstance(encoder_config, dict) or not isinstance(search_config, dict):
        raise ValueError("retrieval.dense requires encoder and search mappings")
    if encoder_config.get("backend") != "sentence_transformers":
        raise ValueError("Only dense encoder backend='sentence_transformers' is supported")
    if search_config.get("backend") != "exact":
        raise ValueError("Only dense search backend='exact' is currently supported")

    encoder = SentenceTransformerEncoder(
        SentenceTransformerEncoderConfig(
            model_name=str(encoder_config["model_name"]),
            revision=(
                str(encoder_config["revision"])
                if encoder_config.get("revision") is not None
                else None
            ),
            batch_size=int(encoder_config.get("batch_size", 32)),
            query_prefix=str(encoder_config.get("query_prefix", "")),
            document_prefix=str(encoder_config.get("document_prefix", "")),
            max_seq_length=(
                int(encoder_config["max_seq_length"])
                if encoder_config.get("max_seq_length") is not None
                else None
            ),
            normalize_embeddings=bool(encoder_config.get("normalize_embeddings", False)),
            trust_remote_code=bool(encoder_config.get("trust_remote_code", False)),
            device=_dense_device(runtime),
        )
    )
    search = ExactVectorSearch(
        metric=str(search_config.get("metric", "cosine")),
        query_batch_size=int(search_config.get("query_batch_size", 64)),
    )
    cache = None
    if bool(cache_config.get("enabled", True)):
        cache = EmbeddingCache(Path(str(cache_config.get("directory", "embeddings"))))
    return DenseRetriever(encoder=encoder, search=search, cache=cache)
