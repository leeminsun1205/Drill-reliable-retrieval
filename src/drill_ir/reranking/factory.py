"""Build reranking components from explicit configuration."""

from typing import Any

from drill_ir.preprocessing import (
    ArticleTruncationChunker,
    DiemChunker,
    KhoanChunker,
    TokenWindowChunker,
    TokenWindowChunkerConfig,
)
from drill_ir.retrieval.factory import _dense_device

from .cross_encoder import CrossEncoderReranker, HFCrossEncoderConfig, HFCrossEncoderScorer


def build_reranker(
    config: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> CrossEncoderReranker:
    if not bool(config.get("enabled", True)):
        raise ValueError("hybrid_reranker requires reranking.enabled=true")
    if config.get("backend", "hf_sequence_classification") != "hf_sequence_classification":
        raise ValueError("Only reranking backend='hf_sequence_classification' is supported")
    model_name = str(config["model_name"])
    representation = str(config.get("representation", "diem"))
    document_max_tokens = int(config.get("document_max_tokens", 768))
    overlap_tokens = int(config.get("overlap_tokens", 64))
    common = TokenWindowChunkerConfig(
        model_name=model_name,
        revision=str(config["revision"]) if config.get("revision") is not None else None,
        max_tokens=document_max_tokens,
        overlap_tokens=overlap_tokens,
        document_prefix="",
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if representation == "article":
        chunker = ArticleTruncationChunker()
    elif representation == "token_window":
        chunker = TokenWindowChunker(common)
    elif representation == "khoan":
        chunker = KhoanChunker(common)
    elif representation == "diem":
        chunker = DiemChunker(common)
    else:
        raise ValueError(
            "reranking.representation must be 'article', 'token_window', 'khoan', or 'diem'"
        )
    scorer = HFCrossEncoderScorer(
        HFCrossEncoderConfig(
            model_name=model_name,
            revision=common.revision,
            max_length=int(config.get("max_length", 1024)),
            batch_size=int(config.get("batch_size", 8)),
            trust_remote_code=common.trust_remote_code,
            positive_label_index=int(config.get("positive_label_index", 1)),
            fp16=bool(config.get("fp16", True)),
            device=_dense_device(runtime_config or {}),
        )
    )
    return CrossEncoderReranker(
        scorer=scorer,
        chunker=chunker,
        candidate_pool_size=int(config.get("candidate_pool_size", 50)),
        representation=representation,
        candidate_source=str(config.get("candidate_source", "article_rechunk")),
    )
