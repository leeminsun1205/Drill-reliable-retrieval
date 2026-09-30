"""Cross-encoder reranking over article representations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from drill_ir.data.models import Article, Query, RetrievalResult
from drill_ir.preprocessing import ArticleChunker, Chunk
from drill_ir.retrieval.dense import ChunkRetrievalResult


class PairScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class HFCrossEncoderConfig:
    model_name: str
    revision: str | None = None
    max_length: int = 1024
    batch_size: int = 8
    trust_remote_code: bool = False
    positive_label_index: int = 1
    fp16: bool = True
    device: str | list[str] = "auto"


class HFCrossEncoderScorer:
    """Hugging Face sequence classifier with document-only truncation."""

    def __init__(self, config: HFCrossEncoderConfig) -> None:
        if not config.model_name:
            raise ValueError("Reranker model_name cannot be empty")
        if config.max_length <= 0 or config.batch_size <= 0:
            raise ValueError("Reranker max_length and batch_size must be positive")
        if config.positive_label_index < 0:
            raise ValueError("positive_label_index cannot be negative")
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._primary_device = "cpu"
        self._resolved_devices: list[str] = []

    def _resolve_devices(self, torch: Any) -> list[str]:
        if self.config.device == "cpu":
            return ["cpu"]
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.config.device == "auto":
            return [f"cuda:{index}" for index in range(count)] if count else ["cpu"]
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
                f"Requested reranker devices {requested}, but only {count} CUDA device(s) exist"
            )
        return list(requested)

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            import torch

            return torch, self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Cross-encoder reranking requires dense dependencies. "
                "Install with: python -m pip install -e '.[dense]'"
            ) from error
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._resolved_devices = self._resolve_devices(torch)
        self._primary_device = self._resolved_devices[0]
        model.to(self._primary_device)
        if len(self._resolved_devices) > 1:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(device.removeprefix("cuda:")) for device in self._resolved_devices],
            )
        model.eval()
        self._model = model
        return torch, self._tokenizer, self._model

    def _positive_scores(self, logits: Any) -> Any:
        if logits.ndim == 1:
            return logits
        if logits.ndim != 2:
            raise ValueError(f"Unsupported reranker logits shape: {tuple(logits.shape)}")
        if logits.shape[1] == 1:
            return logits[:, 0]
        if self.config.positive_label_index >= logits.shape[1]:
            raise ValueError("positive_label_index exceeds reranker label dimension")
        return logits[:, self.config.positive_label_index]

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        torch, tokenizer, model = self._load()
        scores: list[np.ndarray] = []
        use_fp16 = self.config.fp16 and self._primary_device.startswith("cuda:")
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            encoded = tokenizer(
                [query for query, _ in batch],
                [document for _, document in batch],
                padding=True,
                truncation="only_second",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._primary_device) for key, value in encoded.items()}
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                logits = model(**encoded).logits
            scores.append(
                self._positive_scores(logits).detach().float().cpu().numpy()
            )
        return np.concatenate(scores).astype(np.float32, copy=False)

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "truncation": "only_second",
            "resolved_devices": self._resolved_devices,
        }


class CrossEncoderReranker:
    """Score query-representation pairs and max-aggregate them to article IDs."""

    def __init__(
        self,
        *,
        scorer: PairScorer,
        chunker: ArticleChunker,
        candidate_pool_size: int,
        representation: str,
        candidate_source: str = "article_rechunk",
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("reranking.candidate_pool_size must be positive")
        self.scorer = scorer
        self.chunker = chunker
        self.candidate_pool_size = candidate_pool_size
        self.representation = representation
        self.candidate_source = candidate_source
        self._chunks_by_aid: dict[int, list[Chunk]] = {}
        self._representation_count = 0
        self._retrieved_chunk_pairs = 0

    def fit(self, articles: Sequence[Article]) -> None:
        chunks = self.chunker.chunk_articles(articles)
        grouped: defaultdict[int, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.aid].append(chunk)
        missing = {article.aid for article in articles} - set(grouped)
        if missing:
            raise ValueError(f"Reranker produced no representation for aids: {sorted(missing)[:5]}")
        self._chunks_by_aid = dict(grouped)
        self._representation_count = len(chunks)

    def rerank(
        self,
        queries: Sequence[Query],
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        by_qid = {result.qid: result for result in candidates}
        if len(by_qid) != len(candidates):
            raise ValueError("First-stage candidates contain duplicate query IDs")
        reranked = []
        try:
            from tqdm.auto import tqdm

            query_iterator = tqdm(queries, desc="Reranking queries")
        except ImportError:
            query_iterator = queries
        for query in query_iterator:
            candidate = by_qid.get(query.qid)
            if candidate is None:
                raise ValueError(f"Missing first-stage candidates for qid={query.qid}")
            candidate_aids = candidate.aids[: self.candidate_pool_size]
            first_stage_rank = {aid: rank for rank, aid in enumerate(candidate_aids)}
            pairs: list[tuple[str, str]] = []
            pair_aids: list[int] = []
            for aid in candidate_aids:
                chunks = self._chunks_by_aid.get(aid)
                if not chunks:
                    raise ValueError(f"No reranking representation for candidate aid={aid}")
                for chunk in chunks:
                    pairs.append((query.question, chunk.text))
                    pair_aids.append(aid)
            pair_scores = self.scorer.score(pairs)
            if len(pair_scores) != len(pair_aids):
                raise ValueError("Reranker returned an unexpected number of pair scores")
            aid_scores: dict[int, float] = {}
            for aid, score in zip(pair_aids, pair_scores):
                aid_scores[aid] = max(aid_scores.get(aid, -float("inf")), float(score))
            ordered = sorted(
                aid_scores.items(),
                key=lambda item: (-item[1], first_stage_rank[item[0]], item[0]),
            )
            reranked.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(aid for aid, _ in ordered),
                    scores=tuple(score for _, score in ordered),
                )
            )
        return reranked

    def rerank_chunks(
        self,
        queries: Sequence[Query],
        candidates: Sequence[ChunkRetrievalResult],
    ) -> list[RetrievalResult]:
        """Score retrieved evidence directly, then max-aggregate scores by aid."""
        by_qid = {result.qid: result for result in candidates}
        if len(by_qid) != len(candidates):
            raise ValueError("Retrieved chunk candidates contain duplicate query IDs")
        reranked = []
        for query in queries:
            candidate = by_qid.get(query.qid)
            if candidate is None:
                raise ValueError(f"Missing retrieved chunks for qid={query.qid}")
            chunks = candidate.chunks[: self.candidate_pool_size]
            pairs = [(query.question, chunk.text) for chunk in chunks]
            pair_scores = self.scorer.score(pairs)
            if len(pair_scores) != len(chunks):
                raise ValueError("Reranker returned an unexpected number of pair scores")
            aid_scores: dict[int, float] = {}
            best_chunk_rank: dict[int, int] = {}
            for rank, (chunk, score) in enumerate(zip(chunks, pair_scores)):
                value = float(score)
                if value > aid_scores.get(chunk.aid, -float("inf")):
                    aid_scores[chunk.aid] = value
                    best_chunk_rank[chunk.aid] = rank
            ordered = sorted(
                aid_scores.items(),
                key=lambda item: (-item[1], best_chunk_rank[item[0]], item[0]),
            )
            reranked.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(aid for aid, _ in ordered),
                    scores=tuple(score for _, score in ordered),
                )
            )
            self._retrieved_chunk_pairs += len(chunks)
        return reranked

    def run_metadata(self) -> dict[str, Any]:
        scorer_metadata = getattr(self.scorer, "metadata", None)
        return {
            "backend": "hf_sequence_classification",
            "representation": self.representation,
            "candidate_source": self.candidate_source,
            "candidate_pool_size": self.candidate_pool_size,
            "aggregation": "max",
            "candidate_articles": len(self._chunks_by_aid),
            "representations": self._representation_count,
            "retrieved_chunk_pairs": self._retrieved_chunk_pairs,
            "chunking": getattr(self.chunker, "last_stats", {}),
            "scorer": scorer_metadata() if callable(scorer_metadata) else {},
        }
