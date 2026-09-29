"""Deterministic article representations for dense retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Any, Protocol, Sequence

from drill_ir.data.models import Article


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    aid: int
    law_id: str
    chunk_index: int
    chunk_type: str
    text: str
    char_start: int
    char_end: int
    token_count: int | None
    parse_status: str = "not_applicable"
    structure_path: tuple[str, ...] = ()
    source_spans: tuple[tuple[int, int], ...] = ()


class ArticleChunker(Protocol):
    @property
    def cache_key(self) -> str: ...

    def chunk_articles(self, articles: Sequence[Article]) -> list[Chunk]: ...


class ArticleTruncationChunker:
    """One representation per article; truncation remains inside the encoder."""

    @property
    def cache_key(self) -> str:
        return '{"method":"article_truncate","version":1}'

    def chunk_articles(self, articles: Sequence[Article]) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{article.aid}:article:0",
                aid=article.aid,
                law_id=article.law_id,
                chunk_index=0,
                chunk_type="article",
                text=article.text,
                char_start=0,
                char_end=len(article.text),
                token_count=None,
                source_spans=((0, len(article.text)),),
            )
            for article in articles
        ]


@dataclass(frozen=True, slots=True)
class TokenWindowChunkerConfig:
    model_name: str
    revision: str | None = None
    max_tokens: int = 512
    overlap_tokens: int = 64
    document_prefix: str = ""
    trust_remote_code: bool = False


class TokenWindowChunker:
    """Split articles with the exact model tokenizer and raw character offsets."""

    def __init__(self, config: TokenWindowChunkerConfig, *, tokenizer: Any = None) -> None:
        if not config.model_name:
            raise ValueError("Token-window chunking requires a model_name")
        if config.max_tokens <= 0:
            raise ValueError("chunking.max_tokens must be positive")
        if config.overlap_tokens < 0:
            raise ValueError("chunking.overlap_tokens cannot be negative")
        self.config = config
        self._tokenizer = tokenizer
        self.last_stats: dict[str, Any] = {}

    @property
    def cache_key(self) -> str:
        return json.dumps(
            {"method": "token_window", "version": 1, **asdict(self.config)},
            sort_keys=True,
            separators=(",", ":"),
        )

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Token-window chunking requires dense dependencies. "
                "Install with: python -m pip install -e '.[dense]'"
            ) from error
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        return self._tokenizer

    def _token_budget(self, tokenizer: Any) -> tuple[int, int]:
        prefix_ids = tokenizer(
            self.config.document_prefix,
            add_special_tokens=False,
        )["input_ids"]
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        overhead = len(prefix_ids) + special_tokens
        budget = self.config.max_tokens - overhead
        if budget <= 0:
            raise ValueError("Chunk token budget is exhausted by prefix and special tokens")
        if self.config.overlap_tokens >= budget:
            raise ValueError("chunking.overlap_tokens must be smaller than content token budget")
        return budget, overhead

    def chunk_articles(self, articles: Sequence[Article]) -> list[Chunk]:
        tokenizer = self._load_tokenizer()
        if not getattr(tokenizer, "is_fast", True):
            raise ValueError("Token-window chunking requires a fast tokenizer with offsets")
        content_budget, token_overhead = self._token_budget(tokenizer)
        chunks: list[Chunk] = []
        per_aid: Counter[int] = Counter()
        maximum_encoded_tokens = 0
        for article in articles:
            encoded = tokenizer(
                article.text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
                verbose=False,
            )
            token_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if len(token_ids) != len(offsets):
                raise ValueError("Tokenizer IDs and offsets have different lengths")
            if not token_ids:
                continue
            index = 0
            token_start = 0
            while token_start < len(token_ids):
                token_end = min(token_start + content_budget, len(token_ids))
                while True:
                    window_offsets = [
                        (int(start), int(end))
                        for start, end in offsets[token_start:token_end]
                        if int(end) > int(start)
                    ]
                    if not window_offsets:
                        raise ValueError("Tokenizer window has no usable character offsets")
                    char_start = window_offsets[0][0]
                    char_end = window_offsets[-1][1]
                    text = article.text[char_start:char_end]
                    encoded_tokens = len(
                        tokenizer(
                            self.config.document_prefix + text,
                            add_special_tokens=True,
                            truncation=False,
                        )["input_ids"]
                    )
                    if encoded_tokens <= self.config.max_tokens:
                        break
                    token_end -= max(1, encoded_tokens - self.config.max_tokens)
                    if token_end <= token_start:
                        raise ValueError("Cannot create a non-empty chunk within max_tokens")
                chunks.append(
                    Chunk(
                        chunk_id=f"{article.aid}:token_window:{index}",
                        aid=article.aid,
                        law_id=article.law_id,
                        chunk_index=index,
                        chunk_type="token_window",
                        text=text,
                        char_start=char_start,
                        char_end=char_end,
                        token_count=encoded_tokens,
                        source_spans=((char_start, char_end),),
                    )
                )
                per_aid[article.aid] += 1
                maximum_encoded_tokens = max(maximum_encoded_tokens, encoded_tokens)
                index += 1
                if token_end == len(token_ids):
                    break
                next_start = token_end - self.config.overlap_tokens
                if next_start <= token_start:
                    raise ValueError("Chunking made no forward progress")
                token_start = next_start
        self.last_stats = {
            "method": "token_window",
            "articles": len(articles),
            "chunks": len(chunks),
            "content_token_budget": content_budget,
            "prefix_and_special_tokens": token_overhead,
            "max_encoded_tokens_observed": maximum_encoded_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "max_chunks_per_article": max(per_aid.values(), default=0),
        }
        return chunks
