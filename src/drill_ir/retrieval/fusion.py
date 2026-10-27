"""Deterministic rank fusion for independently scored retrievers."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from drill_ir.data.models import Article, Query, RetrievalResult

from .base import Retriever
from .dense import ChunkRetrievalResult


class WeightedRRFRetriever:
    """Fuse sparse and dense ranks with weighted reciprocal rank fusion."""

    def __init__(
        self,
        *,
        sparse: Retriever,
        dense: Retriever,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
        sparse_top_k: int = 100,
        dense_top_k: int = 100,
        include_zero_score_sparse: bool = False,
    ) -> None:
        if sparse_weight < 0 or dense_weight < 0:
            raise ValueError("WRRF weights cannot be negative")
        if sparse_weight == 0 and dense_weight == 0:
            raise ValueError("At least one WRRF weight must be positive")
        if rrf_k <= 0:
            raise ValueError("WRRF rrf_k must be positive")
        if sparse_top_k <= 0 or dense_top_k <= 0:
            raise ValueError("WRRF component top-k values must be positive")
        self.sparse = sparse
        self.dense = dense
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.include_zero_score_sparse = include_zero_score_sparse
        self._component_results: dict[str, list[RetrievalResult]] = {}

    def fit(self, articles: Sequence[Article]) -> None:
        self.sparse.fit(articles)
        self.dense.fit(articles)

    @staticmethod
    def _by_qid(
        results: Sequence[RetrievalResult],
        *,
        component: str,
    ) -> dict[int, RetrievalResult]:
        indexed = {result.qid: result for result in results}
        if len(indexed) != len(results):
            raise ValueError(f"{component} retriever returned duplicate query IDs")
        return indexed

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        sparse_results = self.sparse.retrieve(queries, top_k=self.sparse_top_k)
        dense_results = self.dense.retrieve(queries, top_k=self.dense_top_k)
        sparse_by_qid = self._by_qid(sparse_results, component="sparse")
        dense_by_qid = self._by_qid(dense_results, component="dense")
        expected_qids = {query.qid for query in queries}
        if set(sparse_by_qid) != expected_qids or set(dense_by_qid) != expected_qids:
            raise ValueError("Hybrid component retrievers must return every query exactly once")

        fused_results = []
        for query in queries:
            scores: defaultdict[int, float] = defaultdict(float)
            for component, result, weight in (
                ("sparse", sparse_by_qid[query.qid], self.sparse_weight),
                ("dense", dense_by_qid[query.qid], self.dense_weight),
            ):
                if len(result.aids) != len(result.scores):
                    raise ValueError(f"{component} aids and scores must have equal lengths")
                seen: set[int] = set()
                for rank, (aid, raw_score) in enumerate(
                    zip(result.aids, result.scores), start=1
                ):
                    if aid in seen:
                        continue
                    seen.add(aid)
                    if (
                        component == "sparse"
                        and not self.include_zero_score_sparse
                        and raw_score <= 0
                    ):
                        continue
                    scores[aid] += weight / (self.rrf_k + rank)
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            fused_results.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(aid for aid, _ in ordered),
                    scores=tuple(float(score) for _, score in ordered),
                )
            )
        self._component_results = {
            "sparse": sparse_results,
            "dense": dense_results,
        }
        return fused_results

    def component_results(self) -> dict[str, list[RetrievalResult]]:
        return self._component_results

    def feedback_records(self) -> dict[str, list[dict[str, object]]]:
        records: dict[str, list[dict[str, object]]] = {}
        for name, retriever in (("sparse", self.sparse), ("dense", self.dense)):
            getter = getattr(retriever, "feedback_records", None)
            if callable(getter):
                values = getter()
                if values:
                    records[name] = values
        return records

    def close(self) -> None:
        for retriever in (self.sparse, self.dense):
            close = getattr(retriever, "close", None)
            if callable(close):
                close()

    def retrieve_chunks(
        self,
        queries: Sequence[Query],
        *,
        top_k: int,
        dense_depth: int,
    ) -> list[ChunkRetrievalResult]:
        """Fuse dense chunk ranks with their parent articles' sparse ranks."""
        if top_k <= 0 or dense_depth < top_k:
            raise ValueError("Chunk fusion requires dense_depth >= top_k > 0")
        retrieve_chunks = getattr(self.dense, "retrieve_chunks", None)
        if not callable(retrieve_chunks):
            raise TypeError("Hybrid dense retriever does not expose chunk candidates")
        sparse_results = self._component_results.get("sparse")
        if sparse_results is None:
            raise RuntimeError("retrieve() must be called before retrieve_chunks()")
        sparse_by_qid = self._by_qid(sparse_results, component="sparse")
        dense_chunks = retrieve_chunks(queries, top_k=dense_depth)
        expected_qids = {query.qid for query in queries}
        if set(sparse_by_qid) != expected_qids:
            raise ValueError(
                "Chunk fusion sparse state does not match the requested query batch; "
                "call retrieve() immediately before retrieve_chunks() for the same queries"
            )
        dense_chunk_by_qid = {result.qid: result for result in dense_chunks}
        if len(dense_chunk_by_qid) != len(dense_chunks):
            raise ValueError("Dense chunk retriever returned duplicate query IDs")
        if set(dense_chunk_by_qid) != expected_qids:
            raise ValueError("Dense chunk retriever must return every query exactly once")
        fused: list[ChunkRetrievalResult] = []
        for query in queries:
            result = dense_chunk_by_qid[query.qid]
            sparse = sparse_by_qid[result.qid]
            sparse_ranks = {
                aid: rank
                for rank, (aid, score) in enumerate(
                    zip(sparse.aids, sparse.scores), start=1
                )
                if self.include_zero_score_sparse or score > 0
            }
            ranked = []
            for dense_rank, (chunk, raw_score) in enumerate(
                zip(result.chunks, result.scores), start=1
            ):
                score = self.dense_weight / (self.rrf_k + dense_rank)
                sparse_rank = sparse_ranks.get(chunk.aid)
                if sparse_rank is not None:
                    score += self.sparse_weight / (self.rrf_k + sparse_rank)
                ranked.append((chunk, float(score), dense_rank, raw_score))
            ranked.sort(key=lambda item: (-item[1], item[2], item[0].chunk_id))
            selected = ranked[:top_k]
            fused.append(
                ChunkRetrievalResult(
                    qid=result.qid,
                    chunks=tuple(item[0] for item in selected),
                    scores=tuple(item[1] for item in selected),
                )
            )
        return fused

    def run_metadata(self) -> dict[str, object]:
        sparse_metadata = getattr(self.sparse, "run_metadata", None)
        dense_metadata = getattr(self.dense, "run_metadata", None)
        return {
            "fusion": "weighted_rrf",
            "rrf_k": self.rrf_k,
            "sparse_weight": self.sparse_weight,
            "dense_weight": self.dense_weight,
            "sparse_top_k": self.sparse_top_k,
            "dense_top_k": self.dense_top_k,
            "include_zero_score_sparse": self.include_zero_score_sparse,
            "sparse": sparse_metadata() if callable(sparse_metadata) else {},
            "dense": dense_metadata() if callable(dense_metadata) else {},
        }


class MultiDenseWeightedRRFRetriever:
    """Fuse sparse retrieval with multiple independently configured dense models."""

    def __init__(
        self,
        *,
        sparse: Retriever,
        dense_retrievers: Mapping[str, Retriever],
        sparse_weight: float,
        dense_weights: Mapping[str, float],
        rrf_k: int = 60,
        sparse_top_k: int = 100,
        dense_top_k: Mapping[str, int] | None = None,
        include_zero_score_sparse: bool = False,
    ) -> None:
        if not dense_retrievers:
            raise ValueError("Multi-dense WRRF requires at least one dense component")
        if set(dense_retrievers) != set(dense_weights):
            raise ValueError("Dense retrievers and weights must have identical names")
        if sparse_weight < 0 or any(weight < 0 for weight in dense_weights.values()):
            raise ValueError("WRRF weights cannot be negative")
        if sparse_weight == 0 and not any(dense_weights.values()):
            raise ValueError("At least one WRRF weight must be positive")
        if rrf_k <= 0 or sparse_top_k <= 0:
            raise ValueError("WRRF depths and rrf_k must be positive")
        depths = dict(dense_top_k or {name: 100 for name in dense_retrievers})
        if set(depths) != set(dense_retrievers) or any(depth <= 0 for depth in depths.values()):
            raise ValueError("Every dense component requires a positive top_k")
        self.sparse = sparse
        self.dense_retrievers = dict(dense_retrievers)
        self.dense_weights = dict(dense_weights)
        self.dense_top_k = depths
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.sparse_top_k = sparse_top_k
        self.include_zero_score_sparse = include_zero_score_sparse
        self._component_results: dict[str, list[RetrievalResult]] = {}

    def fit(self, articles: Sequence[Article]) -> None:
        self.sparse.fit(articles)
        for retriever in self.dense_retrievers.values():
            retriever.fit(articles)

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        components = {
            "sparse": self.sparse.retrieve(queries, top_k=self.sparse_top_k),
            **{
                f"dense:{name}": retriever.retrieve(
                    queries, top_k=self.dense_top_k[name]
                )
                for name, retriever in self.dense_retrievers.items()
            },
        }
        indexed = {
            name: WeightedRRFRetriever._by_qid(results, component=name)
            for name, results in components.items()
        }
        expected_qids = {query.qid for query in queries}
        if any(set(by_qid) != expected_qids for by_qid in indexed.values()):
            raise ValueError("Hybrid component retrievers must return every query exactly once")
        fused: list[RetrievalResult] = []
        for query in queries:
            scores: defaultdict[int, float] = defaultdict(float)
            weighted = [("sparse", self.sparse_weight)] + [
                (f"dense:{name}", weight)
                for name, weight in self.dense_weights.items()
            ]
            for component, weight in weighted:
                result = indexed[component][query.qid]
                if len(result.aids) != len(result.scores):
                    raise ValueError(f"{component} aids and scores must have equal lengths")
                seen: set[int] = set()
                for rank, (aid, raw_score) in enumerate(
                    zip(result.aids, result.scores), start=1
                ):
                    if aid in seen:
                        continue
                    seen.add(aid)
                    if (
                        component == "sparse"
                        and not self.include_zero_score_sparse
                        and raw_score <= 0
                    ):
                        continue
                    scores[aid] += weight / (self.rrf_k + rank)
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            fused.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(aid for aid, _ in ordered),
                    scores=tuple(float(score) for _, score in ordered),
                )
            )
        self._component_results = components
        return fused

    def component_results(self) -> dict[str, list[RetrievalResult]]:
        return self._component_results

    def feedback_records(self) -> dict[str, list[dict[str, object]]]:
        records: dict[str, list[dict[str, object]]] = {}
        for name, retriever in [("sparse", self.sparse), *self.dense_retrievers.items()]:
            getter = getattr(retriever, "feedback_records", None)
            if callable(getter) and (values := getter()):
                records[name] = values
        return records

    def close(self) -> None:
        for retriever in (self.sparse, *self.dense_retrievers.values()):
            close = getattr(retriever, "close", None)
            if callable(close):
                close()

    def retrieve_chunks(
        self,
        queries: Sequence[Query],
        *,
        top_k: int,
        dense_depth: int,
    ) -> list[ChunkRetrievalResult]:
        if top_k <= 0 or dense_depth < top_k:
            raise ValueError("Chunk fusion requires dense_depth >= top_k > 0")
        sparse_results = self._component_results.get("sparse")
        if sparse_results is None:
            raise RuntimeError("retrieve() must be called before retrieve_chunks()")
        sparse_by_qid = WeightedRRFRetriever._by_qid(
            sparse_results, component="sparse"
        )
        chunk_components: dict[str, dict[int, ChunkRetrievalResult]] = {}
        for name, retriever in self.dense_retrievers.items():
            retrieve_chunks = getattr(retriever, "retrieve_chunks", None)
            if not callable(retrieve_chunks):
                raise TypeError(f"Dense component {name!r} does not expose chunk candidates")
            results = retrieve_chunks(queries, top_k=dense_depth)
            by_qid = {result.qid: result for result in results}
            if len(by_qid) != len(results):
                raise ValueError(f"Dense component {name!r} returned duplicate query IDs")
            chunk_components[name] = by_qid
        expected_qids = {query.qid for query in queries}
        if set(sparse_by_qid) != expected_qids or any(
            set(results) != expected_qids for results in chunk_components.values()
        ):
            raise ValueError("Chunk retrievers must return every query exactly once")

        fused: list[ChunkRetrievalResult] = []
        for query in queries:
            sparse = sparse_by_qid[query.qid]
            sparse_ranks = {
                aid: rank
                for rank, (aid, score) in enumerate(
                    zip(sparse.aids, sparse.scores), start=1
                )
                if self.include_zero_score_sparse or score > 0
            }
            # Exact text identity safely merges equivalent chunks even when model-specific
            # tokenizers assign different local chunk indices.
            candidates: dict[tuple[int, str], list[object]] = {}
            order = 0
            for name, by_qid in chunk_components.items():
                result = by_qid[query.qid]
                if len(result.chunks) != len(result.scores):
                    raise ValueError(
                        f"Dense component {name!r} chunk/score lengths differ"
                    )
                weight = self.dense_weights[name]
                for rank, chunk in enumerate(result.chunks, start=1):
                    key = (chunk.aid, chunk.text)
                    if key not in candidates:
                        sparse_rank = sparse_ranks.get(chunk.aid)
                        sparse_score = (
                            self.sparse_weight / (self.rrf_k + sparse_rank)
                            if sparse_rank is not None
                            else 0.0
                        )
                        candidates[key] = [chunk, sparse_score, order]
                        order += 1
                    candidates[key][1] = float(candidates[key][1]) + weight / (
                        self.rrf_k + rank
                    )
            ranked = sorted(
                candidates.values(), key=lambda item: (-float(item[1]), int(item[2]))
            )[:top_k]
            fused.append(
                ChunkRetrievalResult(
                    qid=query.qid,
                    chunks=tuple(item[0] for item in ranked),
                    scores=tuple(float(item[1]) for item in ranked),
                )
            )
        return fused

    def run_metadata(self) -> dict[str, object]:
        return {
            "fusion": "multi_dense_weighted_rrf",
            "rrf_k": self.rrf_k,
            "sparse_weight": self.sparse_weight,
            "sparse_top_k": self.sparse_top_k,
            "include_zero_score_sparse": self.include_zero_score_sparse,
            "dense_components": {
                name: {
                    "weight": self.dense_weights[name],
                    "top_k": self.dense_top_k[name],
                    "retriever": (
                        retriever.run_metadata()
                        if callable(getattr(retriever, "run_metadata", None))
                        else {}
                    ),
                }
                for name, retriever in self.dense_retrievers.items()
            },
        }
