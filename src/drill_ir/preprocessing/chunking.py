"""Deterministic article representations for dense retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Protocol, Sequence

from drill_ir.data.models import Article
from drill_ir.structure import parse_clause_spans, parse_point_spans


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
            {"method": self.method, "version": 1, **self._cache_parameters()},
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def method(self) -> str:
        return "token_window"

    def _cache_parameters(self) -> dict[str, Any]:
        return asdict(self.config)

    def _choose_token_end(
        self,
        text: str,
        offsets: Sequence[tuple[int, int]],
        token_start: int,
        candidate_end: int,
    ) -> int:
        del text, offsets
        return candidate_end

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
                token_end = self._choose_token_end(
                    article.text,
                    offsets,
                    token_start,
                    token_end,
                )
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
                        chunk_id=f"{article.aid}:{self.method}:{index}",
                        aid=article.aid,
                        law_id=article.law_id,
                        chunk_index=index,
                        chunk_type=self.method,
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
            "method": self.method,
            "articles": len(articles),
            "chunks": len(chunks),
            "content_token_budget": content_budget,
            "prefix_and_special_tokens": token_overhead,
            "max_encoded_tokens_observed": maximum_encoded_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "max_chunks_per_article": max(per_aid.values(), default=0),
        }
        return chunks


@dataclass(frozen=True, slots=True)
class RecursiveTokenChunkerConfig(TokenWindowChunkerConfig):
    minimum_fill_ratio: float = 0.5


class RecursiveTokenChunker(TokenWindowChunker):
    """Prefer paragraph, line, sentence, then whitespace window boundaries."""

    _SEPARATORS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("paragraph", re.compile(r"\n[ \t]*\n+")),
        ("line", re.compile(r"\n+")),
        ("sentence", re.compile(r"[.!?…][\"'”’\)\]]*\s+")),
        ("whitespace", re.compile(r"\s+")),
    )

    def __init__(
        self,
        config: RecursiveTokenChunkerConfig,
        *,
        tokenizer: Any = None,
    ) -> None:
        if not 0 < config.minimum_fill_ratio <= 1:
            raise ValueError("chunking.minimum_fill_ratio must be in (0, 1]")
        super().__init__(config, tokenizer=tokenizer)
        self.config = config
        self._boundary_counts: Counter[str] = Counter()

    @property
    def method(self) -> str:
        return "recursive_token"

    def _choose_token_end(
        self,
        text: str,
        offsets: Sequence[tuple[int, int]],
        token_start: int,
        candidate_end: int,
    ) -> int:
        if candidate_end >= len(offsets):
            self._boundary_counts["article_end"] += 1
            return candidate_end
        usable = [
            (index, int(start), int(end))
            for index, (start, end) in enumerate(
                offsets[token_start:candidate_end], start=token_start
            )
            if int(end) > int(start)
        ]
        if not usable:
            self._boundary_counts["hard_token"] += 1
            return candidate_end
        char_start = usable[0][1]
        char_end = usable[-1][2]
        minimum_tokens = max(
            1,
            int((candidate_end - token_start) * self.config.minimum_fill_ratio),
        )
        minimum_end = token_start + minimum_tokens
        candidate_text = text[char_start:char_end]
        for name, pattern in self._SEPARATORS:
            for match in reversed(list(pattern.finditer(candidate_text))):
                boundary = char_start + match.end()
                selected_end = token_start
                for index, _, token_char_end in usable:
                    if token_char_end <= boundary:
                        selected_end = index + 1
                    else:
                        break
                if minimum_end <= selected_end <= candidate_end:
                    self._boundary_counts[name] += 1
                    return selected_end
        self._boundary_counts["hard_token"] += 1
        return candidate_end

    def chunk_articles(self, articles: Sequence[Article]) -> list[Chunk]:
        self._boundary_counts.clear()
        chunks = super().chunk_articles(articles)
        self.last_stats["minimum_fill_ratio"] = self.config.minimum_fill_ratio
        self.last_stats["boundary_counts"] = dict(sorted(self._boundary_counts.items()))
        return chunks


class KhoanChunker:
    """Use validated numbered-clause spans with token-window fallback."""

    def __init__(self, config: TokenWindowChunkerConfig, *, tokenizer: Any = None) -> None:
        self.config = config
        self._window_chunker = TokenWindowChunker(config, tokenizer=tokenizer)
        self.last_stats: dict[str, Any] = {}

    @property
    def cache_key(self) -> str:
        return json.dumps(
            {
                "method": "khoan",
                "version": 1,
                "parser": "conservative_line_sequence_v1",
                **asdict(self.config),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _window_span(
        self,
        article: Article,
        *,
        char_start: int,
        char_end: int,
        clause_number: int | None,
        parse_status: str,
        first_index: int,
    ) -> list[Chunk]:
        span_text = article.text[char_start:char_end]
        local = Article(aid=article.aid, law_id=article.law_id, text=span_text)
        windows = self._window_chunker.chunk_articles([local])
        chunks: list[Chunk] = []
        for offset, window in enumerate(windows):
            global_start = char_start + window.char_start
            global_end = char_start + window.char_end
            is_fallback_window = len(windows) > 1
            if clause_number is None:
                chunk_type = "article_fallback_window"
                structure_path: tuple[str, ...] = ()
            else:
                chunk_type = "khoan_token_window" if is_fallback_window else "khoan"
                structure_path = (f"khoan:{clause_number}",)
            chunks.append(
                Chunk(
                    chunk_id=f"{article.aid}:khoan:{first_index + offset}",
                    aid=article.aid,
                    law_id=article.law_id,
                    chunk_index=first_index + offset,
                    chunk_type=chunk_type,
                    text=window.text,
                    char_start=global_start,
                    char_end=global_end,
                    token_count=window.token_count,
                    parse_status=parse_status,
                    structure_path=structure_path,
                    source_spans=((global_start, global_end),),
                )
            )
        return chunks

    def chunk_articles(self, articles: Sequence[Article]) -> list[Chunk]:
        chunks: list[Chunk] = []
        statuses: Counter[str] = Counter()
        oversized_clause_windows = 0
        maximum_encoded_tokens = 0
        per_aid: Counter[int] = Counter()
        for article in articles:
            parsed = parse_clause_spans(article.text)
            statuses[parsed.status] += 1
            article_chunks: list[Chunk] = []
            if parsed.status == "parsed":
                next_index = 0
                for span in parsed.spans:
                    span_chunks = self._window_span(
                        article,
                        char_start=span.char_start,
                        char_end=span.char_end,
                        clause_number=span.number,
                        parse_status="parsed",
                        first_index=next_index,
                    )
                    oversized_clause_windows += max(0, len(span_chunks) - 1)
                    article_chunks.extend(span_chunks)
                    next_index += len(span_chunks)
            else:
                article_chunks = self._window_span(
                    article,
                    char_start=0,
                    char_end=len(article.text),
                    clause_number=None,
                    parse_status=parsed.status,
                    first_index=0,
                )
            chunks.extend(article_chunks)
            per_aid[article.aid] = len(article_chunks)
            maximum_encoded_tokens = max(
                maximum_encoded_tokens,
                max((chunk.token_count or 0 for chunk in article_chunks), default=0),
            )
        self.last_stats = {
            "method": "khoan",
            "parser": "conservative_line_sequence_v1",
            "articles": len(articles),
            "chunks": len(chunks),
            "parse_status_counts": dict(sorted(statuses.items())),
            "oversized_clause_extra_windows": oversized_clause_windows,
            "max_encoded_tokens_observed": maximum_encoded_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "max_chunks_per_article": max(per_aid.values(), default=0),
        }
        return chunks


class DiemChunker:
    """Use Điểm text with its Khoản lead-in and conservative fallbacks."""

    def __init__(self, config: TokenWindowChunkerConfig, *, tokenizer: Any = None) -> None:
        self.config = config
        self._window_chunker = TokenWindowChunker(config, tokenizer=tokenizer)
        self.last_stats: dict[str, Any] = {}

    @property
    def cache_key(self) -> str:
        return json.dumps(
            {
                "method": "diem",
                "version": 1,
                "clause_parser": "conservative_line_sequence_v1",
                "point_parser": "conservative_vietnamese_letter_sequence_v1",
                **asdict(self.config),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _raw_span_chunks(
        self,
        article: Article,
        *,
        char_start: int,
        char_end: int,
        first_index: int,
        chunk_type: str,
        parse_status: str,
        structure_path: tuple[str, ...],
    ) -> list[Chunk]:
        local = Article(
            aid=article.aid,
            law_id=article.law_id,
            text=article.text[char_start:char_end],
        )
        windows = self._window_chunker.chunk_articles([local])
        chunks = []
        for offset, window in enumerate(windows):
            global_start = char_start + window.char_start
            global_end = char_start + window.char_end
            window_type = chunk_type if len(windows) == 1 else f"{chunk_type}_token_window"
            chunks.append(
                Chunk(
                    chunk_id=f"{article.aid}:diem:{first_index + offset}",
                    aid=article.aid,
                    law_id=article.law_id,
                    chunk_index=first_index + offset,
                    chunk_type=window_type,
                    text=window.text,
                    char_start=global_start,
                    char_end=global_end,
                    token_count=window.token_count,
                    parse_status=parse_status,
                    structure_path=structure_path,
                    source_spans=((global_start, global_end),),
                )
            )
        return chunks

    def _point_chunks(
        self,
        article: Article,
        *,
        clause_start: int,
        lead_in_end: int,
        point_start: int,
        point_end: int,
        clause_number: int,
        point_letter: str,
        first_index: int,
    ) -> list[Chunk] | None:
        tokenizer = self._window_chunker._load_tokenizer()
        lead_text = article.text[clause_start:lead_in_end]
        point_text = article.text[point_start:point_end]
        encoded = tokenizer(
            point_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
        if not encoded["input_ids"] or len(encoded["input_ids"]) != len(offsets):
            return None
        full_representation = lead_text + point_text
        full_token_count = len(
            tokenizer(
                self.config.document_prefix + full_representation,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        full_source_spans = tuple(
            span
            for span in ((clause_start, lead_in_end), (point_start, point_end))
            if span[1] > span[0]
        )
        if full_token_count <= self.config.max_tokens:
            return [
                Chunk(
                    chunk_id=f"{article.aid}:diem:{first_index}",
                    aid=article.aid,
                    law_id=article.law_id,
                    chunk_index=first_index,
                    chunk_type="diem",
                    text=full_representation,
                    char_start=min(start for start, _ in full_source_spans),
                    char_end=max(end for _, end in full_source_spans),
                    token_count=full_token_count,
                    parse_status="parsed",
                    structure_path=(f"khoan:{clause_number}", f"diem:{point_letter}"),
                    source_spans=full_source_spans,
                )
            ]

        base_tokens = len(
            tokenizer(
                self.config.document_prefix + lead_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        point_budget = self.config.max_tokens - base_tokens
        if point_budget <= self.config.overlap_tokens:
            return None

        chunks: list[Chunk] = []
        token_start = 0
        while token_start < len(offsets):
            token_end = min(token_start + point_budget, len(offsets))
            while True:
                usable = [
                    (int(start), int(end))
                    for start, end in offsets[token_start:token_end]
                    if int(end) > int(start)
                ]
                if not usable:
                    return None
                local_start = usable[0][0]
                local_end = usable[-1][1]
                point_window = point_text[local_start:local_end]
                representation = lead_text + point_window
                token_count = len(
                    tokenizer(
                        self.config.document_prefix + representation,
                        add_special_tokens=True,
                        truncation=False,
                    )["input_ids"]
                )
                if token_count <= self.config.max_tokens:
                    break
                token_end -= max(1, token_count - self.config.max_tokens)
                if token_end <= token_start:
                    return None

            global_point_start = point_start + local_start
            global_point_end = point_start + local_end
            source_spans = tuple(
                span
                for span in (
                    (clause_start, lead_in_end),
                    (global_point_start, global_point_end),
                )
                if span[1] > span[0]
            )
            index = first_index + len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{article.aid}:diem:{index}",
                    aid=article.aid,
                    law_id=article.law_id,
                    chunk_index=index,
                    chunk_type="diem_token_window",
                    text=representation,
                    char_start=min(start for start, _ in source_spans),
                    char_end=max(end for _, end in source_spans),
                    token_count=token_count,
                    parse_status="parsed",
                    structure_path=(f"khoan:{clause_number}", f"diem:{point_letter}"),
                    source_spans=source_spans,
                )
            )
            if token_end == len(offsets):
                break
            next_start = token_end - self.config.overlap_tokens
            if next_start <= token_start:
                return None
            token_start = next_start
        return chunks

    def chunk_articles(self, articles: Sequence[Article]) -> list[Chunk]:
        chunks: list[Chunk] = []
        article_statuses: Counter[str] = Counter()
        point_statuses: Counter[str] = Counter()
        per_aid: Counter[int] = Counter()
        oversized_point_extra_windows = 0
        lead_in_budget_fallback_clauses = 0
        maximum_encoded_tokens = 0
        for article in articles:
            parsed_clauses = parse_clause_spans(article.text)
            article_statuses[parsed_clauses.status] += 1
            article_chunks: list[Chunk] = []
            next_index = 0
            if parsed_clauses.status != "parsed":
                article_chunks = self._raw_span_chunks(
                    article,
                    char_start=0,
                    char_end=len(article.text),
                    first_index=0,
                    chunk_type="article_fallback",
                    parse_status=parsed_clauses.status,
                    structure_path=(),
                )
            else:
                for clause in parsed_clauses.spans:
                    clause_text = article.text[clause.char_start:clause.char_end]
                    parsed_points = parse_point_spans(clause_text)
                    point_statuses[parsed_points.status] += 1
                    clause_chunks: list[Chunk] = []
                    if parsed_points.status == "parsed":
                        lead_in_end = clause.char_start + parsed_points.lead_in_end
                        for point in parsed_points.spans:
                            point_chunks = self._point_chunks(
                                article,
                                clause_start=clause.char_start,
                                lead_in_end=lead_in_end,
                                point_start=clause.char_start + point.char_start,
                                point_end=clause.char_start + point.char_end,
                                clause_number=clause.number,
                                point_letter=point.letter,
                                first_index=next_index + len(clause_chunks),
                            )
                            if point_chunks is None:
                                clause_chunks = []
                                lead_in_budget_fallback_clauses += 1
                                break
                            oversized_point_extra_windows += max(0, len(point_chunks) - 1)
                            clause_chunks.extend(point_chunks)
                    if not clause_chunks:
                        clause_chunks = self._raw_span_chunks(
                            article,
                            char_start=clause.char_start,
                            char_end=clause.char_end,
                            first_index=next_index,
                            chunk_type="khoan_fallback",
                            parse_status=f"point_{parsed_points.status}",
                            structure_path=(f"khoan:{clause.number}",),
                        )
                    article_chunks.extend(clause_chunks)
                    next_index += len(clause_chunks)
            chunks.extend(article_chunks)
            per_aid[article.aid] = len(article_chunks)
            maximum_encoded_tokens = max(
                maximum_encoded_tokens,
                max((chunk.token_count or 0 for chunk in article_chunks), default=0),
            )
        self.last_stats = {
            "method": "diem",
            "clause_parser": "conservative_line_sequence_v1",
            "point_parser": "conservative_vietnamese_letter_sequence_v1",
            "articles": len(articles),
            "chunks": len(chunks),
            "article_parse_status_counts": dict(sorted(article_statuses.items())),
            "point_parse_status_counts": dict(sorted(point_statuses.items())),
            "oversized_point_extra_windows": oversized_point_extra_windows,
            "lead_in_budget_fallback_clauses": lead_in_budget_fallback_clauses,
            "max_encoded_tokens_observed": maximum_encoded_tokens,
            "overlap_tokens": self.config.overlap_tokens,
            "max_chunks_per_article": max(per_aid.values(), default=0),
        }
        return chunks
