"""Build retrieval components from explicit experiment configuration."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from .base import Retriever
from .bm25 import BM25IndexCache, BM25Retriever, BM25SIndexCache, BM25SRetriever
from .dense import (
    DenseChunkCache,
    DenseRetriever,
    EmbeddingCache,
    ExactVectorSearch,
    FaissFlatVectorSearch,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
)
from .fusion import (
    MultiDenseWeightedRRFRetriever,
    WeightedRRFRetriever,
    load_reference_adjacency,
)
from drill_ir.preprocessing import (
    ArticleTruncationChunker,
    DiemChunker,
    KhoanDiemChunker,
    KhoanChunker,
    RecursiveTokenChunker,
    RecursiveTokenChunkerConfig,
    TokenWindowChunker,
    TokenWindowChunkerConfig,
    build_token_processor,
)


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


def _merged_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a dense profile without mutating the experiment config."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merged_mapping(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_retriever(
    method: str,
    retrieval_config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> Retriever:
    """Build one first-stage retriever behind the common Retriever protocol."""
    runtime = runtime_config or {}
    if method == "hybrid":
        hybrid = retrieval_config.get("hybrid")
        if not isinstance(hybrid, dict):
            raise ValueError("retrieval.hybrid must be a YAML mapping")
        if hybrid.get("fusion", "weighted_rrf") != "weighted_rrf":
            raise ValueError("Only hybrid fusion='weighted_rrf' is currently supported")
        neighbor_expansion = hybrid.get("same_law_neighbor_expansion", {})
        if not isinstance(neighbor_expansion, dict):
            raise ValueError(
                "retrieval.hybrid.same_law_neighbor_expansion must be a YAML mapping"
            )
        neighbor_parameters = {
            "neighbor_expansion_enabled": bool(
                neighbor_expansion.get("enabled", False)
            ),
            "neighbor_seed_top_k": int(neighbor_expansion.get("seed_top_k", 5)),
            "neighbor_radius": int(neighbor_expansion.get("radius", 1)),
            "neighbor_max_expansions": int(
                neighbor_expansion.get("max_expansions", 4)
            ),
        }
        reference_expansion = hybrid.get("reference_expansion", {})
        if not isinstance(reference_expansion, dict):
            raise ValueError("retrieval.hybrid.reference_expansion must be a YAML mapping")
        reference_enabled = bool(reference_expansion.get("enabled", False))
        reference_path_value = reference_expansion.get("graph_path")
        reference_scopes = tuple(
            str(value) for value in reference_expansion.get("scopes", ["same_document"])
        )
        if reference_enabled and not reference_path_value:
            raise ValueError(
                "Enabled retrieval.hybrid.reference_expansion requires graph_path"
            )
        reference_adjacency = (
            load_reference_adjacency(
                Path(str(reference_path_value)), scopes=reference_scopes
            )
            if reference_enabled
            else None
        )
        reference_parameters = {
            "reference_adjacency": reference_adjacency,
            "reference_graph_path": str(reference_path_value) if reference_enabled else None,
            "reference_scopes": reference_scopes,
            "reference_seed_top_k": int(reference_expansion.get("seed_top_k", 5)),
            "reference_max_expansions": int(
                reference_expansion.get("max_expansions", 4)
            ),
        }
        dense_components = hybrid.get("dense_components")
        if dense_components is not None:
            if not isinstance(dense_components, list) or not dense_components:
                raise ValueError("retrieval.hybrid.dense_components must be a non-empty list")
            base_dense = retrieval_config.get("dense")
            profiles = retrieval_config.get("dense_profiles", {})
            if not isinstance(base_dense, dict) or not isinstance(profiles, dict):
                raise ValueError("Multi-dense retrieval requires dense and dense_profiles mappings")
            dense_retrievers: dict[str, Retriever] = {}
            dense_weights: dict[str, float] = {}
            dense_depths: dict[str, int] = {}
            for component in dense_components:
                if not isinstance(component, dict):
                    raise ValueError("Each dense component must be a YAML mapping")
                name = str(component.get("name", "")).strip()
                profile_name = str(component.get("profile", name)).strip()
                if not name or name in dense_retrievers:
                    raise ValueError("Dense component names must be non-empty and unique")
                profile = profiles.get(profile_name)
                if not isinstance(profile, dict):
                    raise ValueError(f"Unknown dense profile {profile_name!r}")
                component_config = _merged_mapping(base_dense, profile)
                dense_retrievers[name] = build_retriever(
                    "dense", {**retrieval_config, "dense": component_config}, runtime
                )
                dense_weights[name] = float(component.get("weight", 1.0))
                dense_depths[name] = int(
                    component.get("top_k", hybrid.get("dense_top_k", 100))
                )
            return MultiDenseWeightedRRFRetriever(
                sparse=build_retriever("bm25", retrieval_config, runtime),
                dense_retrievers=dense_retrievers,
                sparse_weight=float(hybrid.get("sparse_weight", 1.0)),
                dense_weights=dense_weights,
                rrf_k=int(hybrid.get("rrf_k", 60)),
                sparse_top_k=int(hybrid.get("sparse_top_k", 100)),
                dense_top_k=dense_depths,
                include_zero_score_sparse=bool(
                    hybrid.get("include_zero_score_sparse", False)
                ),
                chunk_candidate_selection=str(
                    hybrid.get("chunk_candidate_selection", "top_chunks")
                ),
                **neighbor_parameters,
                **reference_parameters,
            )
        return WeightedRRFRetriever(
            sparse=build_retriever("bm25", retrieval_config, runtime),
            dense=build_retriever("dense", retrieval_config, runtime),
            sparse_weight=float(hybrid.get("sparse_weight", 1.0)),
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            sparse_top_k=int(hybrid.get("sparse_top_k", 100)),
            dense_top_k=int(hybrid.get("dense_top_k", 100)),
            include_zero_score_sparse=bool(
                hybrid.get("include_zero_score_sparse", False)
            ),
            chunk_candidate_selection=str(
                hybrid.get("chunk_candidate_selection", "top_chunks")
            ),
            **neighbor_parameters,
            **reference_parameters,
        )
    if method == "bm25":
        params = retrieval_config.get("bm25", {})
        backend = str(params.get("backend", "native"))
        if backend not in {"native", "bm25s"}:
            raise ValueError("retrieval.bm25.backend must be 'native' or 'bm25s'")
        cache_config = params.get("cache", {})
        if not isinstance(cache_config, dict):
            raise ValueError("retrieval.bm25.cache must be a YAML mapping")
        feedback = params.get("feedback", {})
        if not isinstance(feedback, dict):
            raise ValueError("retrieval.bm25.feedback must be a YAML mapping")
        feedback_method = str(feedback.get("method", "none"))
        if feedback_method not in {"none", "rm3"}:
            raise ValueError("BM25 feedback.method must be 'none' or 'rm3'")
        if bool(feedback.get("enabled", False)) and feedback_method != "rm3":
            raise ValueError("Enabled BM25 feedback requires method='rm3'")
        if backend == "bm25s" and bool(feedback.get("enabled", False)):
            raise ValueError("BM25S backend does not currently support RM3 feedback")
        common_parameters = dict(
            k1=float(params.get("k1", 1.2)),
            b=float(params.get("b", 0.75)),
            variant=str(params.get("variant", "bm25")),
            delta=float(params.get("delta", 1.0)),
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
        if backend == "bm25s":
            return BM25SRetriever(
                **common_parameters,
                cache=(
                    BM25SIndexCache(Path(str(cache_config.get("directory", "indexes/bm25s"))))
                    if bool(cache_config.get("enabled", False))
                    else None
                ),
                require_cache_hit=bool(cache_config.get("require_hit", False)),
            )
        return BM25Retriever(
            **common_parameters,
            rm3_enabled=bool(feedback.get("enabled", False)),
            rm3_feedback_documents=int(feedback.get("feedback_documents", 5)),
            rm3_feedback_terms=int(feedback.get("feedback_terms", 10)),
            rm3_original_query_weight=float(
                feedback.get("original_query_weight", 0.7)
            ),
            rm3_dirichlet_mu=float(feedback.get("dirichlet_mu", 1000.0)),
            cache=(
                BM25IndexCache(Path(str(cache_config.get("directory", "indexes/bm25"))))
                if bool(cache_config.get("enabled", False))
                else None
            ),
            require_cache_hit=bool(cache_config.get("require_hit", False)),
        )
    if method != "dense":
        raise ValueError(f"No retriever factory registered for method={method!r}")

    dense = retrieval_config.get("dense")
    if not isinstance(dense, dict):
        raise ValueError("retrieval.dense must be a YAML mapping")
    encoder_config = dense.get("encoder")
    search_config = dense.get("search")
    cache_config = dense.get("cache", {})
    chunk_cache_config = dense.get("chunk_cache", {})
    chunking_config = dense.get("chunking", {"method": "article_truncate"})
    feedback_config = dense.get("feedback", {})
    if not isinstance(encoder_config, dict) or not isinstance(search_config, dict):
        raise ValueError("retrieval.dense requires encoder and search mappings")
    if not isinstance(chunking_config, dict):
        raise ValueError("retrieval.dense.chunking must be a YAML mapping")
    if not isinstance(chunk_cache_config, dict):
        raise ValueError("retrieval.dense.chunk_cache must be a YAML mapping")
    if not isinstance(feedback_config, dict):
        raise ValueError("retrieval.dense.feedback must be a YAML mapping")
    feedback_method = str(feedback_config.get("method", "none"))
    if feedback_method not in {"none", "rocchio"}:
        raise ValueError("Dense feedback.method must be 'none' or 'rocchio'")
    if bool(feedback_config.get("enabled", False)) and feedback_method != "rocchio":
        raise ValueError("Enabled dense feedback requires method='rocchio'")
    if encoder_config.get("backend") != "sentence_transformers":
        raise ValueError("Only dense encoder backend='sentence_transformers' is supported")
    search_backend = str(search_config.get("backend", "exact_numpy"))
    if search_backend not in {"exact", "exact_numpy", "faiss_flat"}:
        raise ValueError(
            "Dense search backend must be 'exact_numpy' (legacy alias 'exact') "
            "or 'faiss_flat'"
        )

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
    search_parameters = {
        "metric": str(search_config.get("metric", "cosine")),
        "query_batch_size": int(search_config.get("query_batch_size", 64)),
    }
    search = (
        FaissFlatVectorSearch(
            **search_parameters,
            device=str(search_config.get("device", "cpu")),
        )
        if search_backend == "faiss_flat"
        else ExactVectorSearch(**search_parameters)
    )
    cache = None
    if bool(cache_config.get("enabled", True)):
        cache = EmbeddingCache(Path(str(cache_config.get("directory", "embeddings"))))
    chunk_cache = None
    if bool(chunk_cache_config.get("enabled", False)):
        chunk_cache = DenseChunkCache(
            Path(str(chunk_cache_config.get("directory", "embeddings/chunks")))
        )
    chunking_method = chunking_config.get("method", "article_truncate")
    if chunking_config.get("aggregation", "max") != "max":
        raise ValueError("Only dense chunk aggregation='max' is currently supported")
    if chunking_method == "article_truncate":
        chunker = ArticleTruncationChunker()
    elif chunking_method in {"token_window", "recursive_token", "khoan", "diem", "khoan_diem"}:
        max_tokens = int(chunking_config.get("max_tokens", 512))
        encoder_max = encoder_config.get("max_seq_length")
        if encoder_max is not None and max_tokens > int(encoder_max):
            raise ValueError("chunking.max_tokens cannot exceed encoder.max_seq_length")
        common_chunking = {
            "model_name": str(
                chunking_config.get("tokenizer_model_name", encoder_config["model_name"])
            ),
            "revision": (
                str(
                    chunking_config.get(
                        "tokenizer_revision", encoder_config.get("revision")
                    )
                )
                if chunking_config.get(
                    "tokenizer_revision", encoder_config.get("revision")
                )
                is not None
                else None
            ),
            "max_tokens": max_tokens,
            "overlap_tokens": int(chunking_config.get("overlap_tokens", 64)),
            "document_prefix": str(
                chunking_config.get(
                    "tokenizer_document_prefix",
                    encoder_config.get("document_prefix", ""),
                )
            ),
            "trust_remote_code": bool(
                chunking_config.get(
                    "tokenizer_trust_remote_code",
                    encoder_config.get("trust_remote_code", False),
                )
            ),
        }
        if chunking_method == "token_window":
            chunker = TokenWindowChunker(TokenWindowChunkerConfig(**common_chunking))
        elif chunking_method == "recursive_token":
            chunker = RecursiveTokenChunker(
                RecursiveTokenChunkerConfig(
                    **common_chunking,
                    minimum_fill_ratio=float(
                        chunking_config.get("minimum_fill_ratio", 0.5)
                    ),
                )
            )
        elif chunking_method == "khoan":
            chunker = KhoanChunker(TokenWindowChunkerConfig(**common_chunking))
        elif chunking_method == "khoan_diem":
            chunker = KhoanDiemChunker(TokenWindowChunkerConfig(**common_chunking))
        else:
            chunker = DiemChunker(TokenWindowChunkerConfig(**common_chunking))
    else:
        raise ValueError(
            "retrieval.dense.chunking.method must be 'article_truncate', "
            "'token_window', 'recursive_token', 'khoan', 'diem', or 'khoan_diem'"
        )
    return DenseRetriever(
        encoder=encoder,
        search=search,
        cache=cache,
        chunk_cache=chunk_cache,
        chunker=chunker,
        rocchio_enabled=bool(feedback_config.get("enabled", False)),
        rocchio_feedback_documents=int(
            feedback_config.get("feedback_documents", 5)
        ),
        rocchio_alpha=float(feedback_config.get("alpha", 1.0)),
        rocchio_beta=float(feedback_config.get("beta", 0.25)),
        rocchio_gamma=float(feedback_config.get("gamma", 0.0)),
        rocchio_negative_documents=int(
            feedback_config.get("negative_documents", 0)
        ),
        rocchio_candidate_depth=int(feedback_config.get("candidate_depth", 20)),
        require_cache_hit=bool(cache_config.get("require_hit", False)),
    )
