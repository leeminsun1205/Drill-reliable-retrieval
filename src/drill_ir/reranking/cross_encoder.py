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
        print(f"[reranker] Loading checkpoint on CPU: {self.config.model_name}", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        print("[reranker] Checkpoint materialized on CPU", flush=True)
        self._resolved_devices = self._resolve_devices(torch)
        self._primary_device = self._resolved_devices[0]
        print(
            f"[reranker] Moving checkpoint to {self._primary_device}; "
            f"resolved_devices={self._resolved_devices}",
            flush=True,
        )
        model.to(self._primary_device)
        if len(self._resolved_devices) > 1:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(device.removeprefix("cuda:")) for device in self._resolved_devices],
            )
        model.eval()
        self._model = model
        print("[reranker] Checkpoint ready for inference", flush=True)
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
        return self.score_with_batch_size(pairs, batch_size=self.config.batch_size)

    def score_with_batch_size(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Score pairs with one temporary mini-batch size using the loaded model."""
        if batch_size <= 0:
            raise ValueError("Reranker batch_size must be positive")
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        torch, tokenizer, model = self._load()
        scores: list[np.ndarray] = []
        use_fp16 = self.config.fp16 and self._primary_device.startswith("cuda:")
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [query for query, _ in batch],
                [document for _, document in batch],
                padding=True,
                truncation="only_second",
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._primary_device) for key, value in encoded.items()}
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_fp16,
                ):
                    logits = model(**encoded).logits
            except BaseException as error:
                memory = ""
                if torch.cuda.is_available():
                    parts = []
                    for index in range(torch.cuda.device_count()):
                        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                        parts.append(
                            f"cuda:{index} free={free_bytes / 2**20:.0f}MiB/"
                            f"{total_bytes / 2**20:.0f}MiB "
                            f"allocated={torch.cuda.memory_allocated(index) / 2**20:.0f}MiB "
                            f"reserved={torch.cuda.memory_reserved(index) / 2**20:.0f}MiB"
                        )
                    memory = "; ".join(parts)
                print(
                    "[reranker] Forward failed: "
                    f"start={start} batch_size={len(batch)} "
                    f"error_type={type(error).__name__} error={error} cuda={memory}",
                    flush=True,
                )
                raise
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
        batching: str = "global_stream",
        outer_pair_batch_size: int = 1024,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("reranking.candidate_pool_size must be positive")
        if batching not in {"global_stream", "per_query"}:
            raise ValueError("reranking.batching must be 'global_stream' or 'per_query'")
        if outer_pair_batch_size <= 0:
            raise ValueError("reranking.outer_pair_batch_size must be positive")
        self.scorer = scorer
        self.chunker = chunker
        self.candidate_pool_size = candidate_pool_size
        self.representation = representation
        self.candidate_source = candidate_source
        self.batching = batching
        self.outer_pair_batch_size = outer_pair_batch_size
        self._chunks_by_aid: dict[int, list[Chunk]] = {}
        self._representation_count = 0
        self._retrieved_chunk_pairs = 0
        self._scored_pairs = 0
        self._scorer_calls = 0

    def _score_global_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        description: str,
    ) -> np.ndarray:
        """Score a continuous pair stream in bounded outer batches."""
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        starts: Any = range(0, len(pairs), self.outer_pair_batch_size)
        try:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc=description, unit="outer-batch")
        except ImportError:
            pass
        score_batches: list[np.ndarray] = []
        for start in starts:
            batch = pairs[start : start + self.outer_pair_batch_size]
            scores = np.asarray(self.scorer.score(batch), dtype=np.float32)
            if len(scores) != len(batch):
                raise ValueError("Reranker returned an unexpected number of pair scores")
            score_batches.append(scores)
            self._scored_pairs += len(batch)
            self._scorer_calls += 1
        return np.concatenate(score_batches).astype(np.float32, copy=False)

    def fit(self, articles: Sequence[Article]) -> None:
        print(
            f"[reranker] Preparing representations for {len(articles):,} articles",
            flush=True,
        )
        chunks = self.chunker.chunk_articles(articles)
        grouped: defaultdict[int, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.aid].append(chunk)
        missing = {article.aid for article in articles} - set(grouped)
        if missing:
            raise ValueError(f"Reranker produced no representation for aids: {sorted(missing)[:5]}")
        self._chunks_by_aid = dict(grouped)
        self._representation_count = len(chunks)
        print(
            f"[reranker] Prepared {self._representation_count:,} representations",
            flush=True,
        )

    def representations_for_aid(self, aid: int) -> tuple[Chunk, ...]:
        """Return fitted representations for training-data inspection/mining."""
        return tuple(self._chunks_by_aid.get(aid, ()))

    def rerank(
        self,
        queries: Sequence[Query],
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        by_qid = {result.qid: result for result in candidates}
        if len(by_qid) != len(candidates):
            raise ValueError("First-stage candidates contain duplicate query IDs")
        if self.batching == "global_stream":
            return self._rerank_global(queries, by_qid)
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
            self._scored_pairs += len(pairs)
            self._scorer_calls += 1
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

    def _rerank_global(
        self,
        queries: Sequence[Query],
        by_qid: dict[int, RetrievalResult],
    ) -> list[RetrievalResult]:
        pairs: list[tuple[str, str]] = []
        plans: list[tuple[Query, dict[int, int], list[int], int, int]] = []
        for query in queries:
            candidate = by_qid.get(query.qid)
            if candidate is None:
                raise ValueError(f"Missing first-stage candidates for qid={query.qid}")
            candidate_aids = candidate.aids[: self.candidate_pool_size]
            first_stage_rank = {aid: rank for rank, aid in enumerate(candidate_aids)}
            pair_aids: list[int] = []
            start = len(pairs)
            for aid in candidate_aids:
                chunks = self._chunks_by_aid.get(aid)
                if not chunks:
                    raise ValueError(f"No reranking representation for candidate aid={aid}")
                for chunk in chunks:
                    pairs.append((query.question, chunk.text))
                    pair_aids.append(aid)
            plans.append((query, first_stage_rank, pair_aids, start, len(pairs)))
        all_scores = self._score_global_pairs(pairs, description="Reranking pair stream")
        reranked: list[RetrievalResult] = []
        for query, first_stage_rank, pair_aids, start, end in plans:
            aid_scores: dict[int, float] = {}
            for aid, score in zip(pair_aids, all_scores[start:end]):
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
        if self.batching == "global_stream":
            return self._rerank_chunks_global(queries, by_qid)
        reranked = []
        try:
            from tqdm.auto import tqdm

            query_iterator = tqdm(queries, desc="Reranking retrieved chunks", unit="query")
        except ImportError:
            query_iterator = queries
        for query in query_iterator:
            candidate = by_qid.get(query.qid)
            if candidate is None:
                raise ValueError(f"Missing retrieved chunks for qid={query.qid}")
            chunks = candidate.chunks[: self.candidate_pool_size]
            pairs = [(query.question, chunk.text) for chunk in chunks]
            pair_scores = self.scorer.score(pairs)
            self._scored_pairs += len(pairs)
            self._scorer_calls += 1
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

    def _rerank_chunks_global(
        self,
        queries: Sequence[Query],
        by_qid: dict[int, ChunkRetrievalResult],
    ) -> list[RetrievalResult]:
        pairs: list[tuple[str, str]] = []
        plans: list[tuple[Query, tuple[Chunk, ...], int, int]] = []
        for query in queries:
            candidate = by_qid.get(query.qid)
            if candidate is None:
                raise ValueError(f"Missing retrieved chunks for qid={query.qid}")
            chunks = candidate.chunks[: self.candidate_pool_size]
            start = len(pairs)
            pairs.extend((query.question, chunk.text) for chunk in chunks)
            plans.append((query, chunks, start, len(pairs)))
        all_scores = self._score_global_pairs(
            pairs, description="Reranking retrieved-chunk pair stream"
        )
        reranked: list[RetrievalResult] = []
        for query, chunks, start, end in plans:
            aid_scores: dict[int, float] = {}
            best_chunk_rank: dict[int, int] = {}
            for rank, (chunk, score) in enumerate(zip(chunks, all_scores[start:end])):
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
            "batching": self.batching,
            "outer_pair_batch_size": self.outer_pair_batch_size,
            "scored_pairs": self._scored_pairs,
            "scorer_calls": self._scorer_calls,
            "aggregation": "max",
            "candidate_articles": len(self._chunks_by_aid),
            "representations": self._representation_count,
            "retrieved_chunk_pairs": self._retrieved_chunk_pairs,
            "chunking": getattr(self.chunker, "last_stats", {}),
            "scorer": scorer_metadata() if callable(scorer_metadata) else {},
        }
