"""Deterministic rank fusion for independently scored retrievers."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from drill_ir.data.models import Article, Query, RetrievalResult
from drill_ir.runtime_resources import cuda_snapshot

from .base import Retriever
from .dense import ChunkRetrievalResult


_CHUNK_CANDIDATE_SELECTIONS = {"top_chunks", "best_per_aid"}
_REFERENCE_SCOPES = {"same_document", "cross_document"}


def load_reference_adjacency(
    path: Path, *, scopes: Sequence[str]
) -> dict[int, list[dict[str, object]]]:
    """Load verified eligible graph edges while preserving artifact order."""
    requested = tuple(dict.fromkeys(scopes))
    unknown = set(requested) - _REFERENCE_SCOPES
    if not requested or unknown:
        raise ValueError(
            "reference expansion scopes must be a non-empty subset of "
            f"{sorted(_REFERENCE_SCOPES)}; unknown={sorted(unknown)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Reference graph does not exist: {path}")
    adjacency: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[int, int, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid reference graph JSON at {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Reference graph row must be an object at {path}:{line_number}"
                )
            if not bool(record.get("candidate_eligible", False)):
                raise ValueError(
                    "Reference edge artifact must contain eligible edges only; "
                    f"found ineligible row at {path}:{line_number}"
                )
            scope = str(record.get("scope", ""))
            if scope not in _REFERENCE_SCOPES:
                raise ValueError(
                    f"Unknown reference scope {scope!r} at {path}:{line_number}"
                )
            source_aid = int(record["source_aid"])
            target_aid = int(record["target_aid"])
            if source_aid <= 0 or target_aid <= 0:
                raise ValueError(
                    f"Reference graph aids must be positive at {path}:{line_number}"
                )
            if scope not in requested:
                continue
            key = (source_aid, target_aid, scope)
            if key in seen:
                continue
            seen.add(key)
            adjacency[source_aid].append(
                {
                    "target_aid": target_aid,
                    "scope": scope,
                    "target_law_id": str(record.get("target_law_id", "")),
                }
            )
    return dict(adjacency)


def _validate_chunk_candidate_selection(value: str) -> str:
    if value not in _CHUNK_CANDIDATE_SELECTIONS:
        choices = ", ".join(sorted(_CHUNK_CANDIDATE_SELECTIONS))
        raise ValueError(f"chunk_candidate_selection must be one of: {choices}")
    return value


def _select_chunk_candidates(
    ranked: Sequence[Sequence[object]],
    *,
    top_k: int,
    selection: str,
) -> list[Sequence[object]]:
    """Select a fixed-size chunk pool, optionally limiting it to one chunk per aid."""
    if selection == "top_chunks":
        return list(ranked[:top_k])
    selected: list[Sequence[object]] = []
    seen_aids: set[int] = set()
    for item in ranked:
        chunk = item[0]
        aid = int(getattr(chunk, "aid"))
        if aid in seen_aids:
            continue
        seen_aids.add(aid)
        selected.append(item)
        if len(selected) == top_k:
            break
    return selected


def _same_law_neighbors(
    articles: Sequence[Article], *, radius: int
) -> dict[int, list[tuple[int, int]]]:
    by_law: dict[str, list[int]] = {}
    for article in articles:
        by_law.setdefault(article.law_id, []).append(article.aid)
    neighbors: dict[int, list[tuple[int, int]]] = {}
    for aids in by_law.values():
        for position, aid in enumerate(aids):
            values: list[tuple[int, int]] = []
            for distance in range(1, radius + 1):
                if position - distance >= 0:
                    values.append((aids[position - distance], -distance))
                if position + distance < len(aids):
                    values.append((aids[position + distance], distance))
            neighbors[aid] = values
    return neighbors


def _expand_with_available_neighbors(
    ranked: Sequence[Sequence[object]],
    selected: Sequence[Sequence[object]],
    *,
    qid: int,
    top_k: int,
    neighbors: Mapping[int, Sequence[tuple[int, int]]],
    seed_top_k: int,
    max_expansions: int,
) -> tuple[list[Sequence[object]], list[dict[str, int]]]:
    """Replace tail chunks with available same-law neighbors at a fixed budget."""
    best_by_aid: dict[int, Sequence[object]] = {}
    for item in ranked:
        aid = int(getattr(item[0], "aid"))
        best_by_aid.setdefault(aid, item)
    selected_aids = {int(getattr(item[0], "aid")) for item in selected}
    seed_aids: list[int] = []
    for item in selected:
        aid = int(getattr(item[0], "aid"))
        if aid not in seed_aids:
            seed_aids.append(aid)
        if len(seed_aids) == seed_top_k:
            break
    expansions: list[Sequence[object]] = []
    records: list[dict[str, int]] = []
    for seed_aid in seed_aids:
        for neighbor_aid, signed_distance in neighbors.get(seed_aid, ()):
            candidate = best_by_aid.get(neighbor_aid)
            if candidate is None or neighbor_aid in selected_aids:
                continue
            selected_aids.add(neighbor_aid)
            expansions.append(candidate)
            records.append(
                {
                    "qid": qid,
                    "seed_aid": seed_aid,
                    "neighbor_aid": neighbor_aid,
                    "signed_distance": signed_distance,
                }
            )
            if len(expansions) == max_expansions:
                break
        if len(expansions) == max_expansions:
            break
    if not expansions:
        return list(selected), records
    return [*selected[: top_k - len(expansions)], *expansions], records


def _expand_with_available_references(
    ranked: Sequence[Sequence[object]],
    selected: Sequence[Sequence[object]],
    *,
    qid: int,
    top_k: int,
    references: Mapping[int, Sequence[Mapping[str, object]]],
    seed_top_k: int,
    max_expansions: int,
) -> tuple[list[Sequence[object]], list[dict[str, object]]]:
    """Replace tail chunks with available explicit-reference targets."""
    best_by_aid: dict[int, Sequence[object]] = {}
    rank_by_aid: dict[int, int] = {}
    for rank, item in enumerate(ranked, start=1):
        aid = int(getattr(item[0], "aid"))
        best_by_aid.setdefault(aid, item)
        rank_by_aid.setdefault(aid, rank)
    selected_aids = {int(getattr(item[0], "aid")) for item in selected}
    seed_aids: list[int] = []
    for item in selected:
        aid = int(getattr(item[0], "aid"))
        if aid not in seed_aids:
            seed_aids.append(aid)
        if len(seed_aids) == seed_top_k:
            break
    proposals: list[
        tuple[int, int, int, int, Sequence[object], Mapping[str, object]]
    ] = []
    proposed_aids: set[int] = set()
    for seed_index, seed_aid in enumerate(seed_aids):
        for edge_index, edge in enumerate(references.get(seed_aid, ())):
            target_aid = int(edge["target_aid"])
            candidate = best_by_aid.get(target_aid)
            if (
                candidate is None
                or target_aid in selected_aids
                or target_aid in proposed_aids
            ):
                continue
            proposed_aids.add(target_aid)
            proposals.append(
                (
                    rank_by_aid[target_aid],
                    seed_index,
                    edge_index,
                    target_aid,
                    candidate,
                    {"seed_aid": seed_aid, **edge},
                )
            )
    proposals.sort(key=lambda proposal: proposal[:4])
    chosen = proposals[:max_expansions]
    expansions = [proposal[4] for proposal in chosen]
    records = [
        {
            "qid": qid,
            "seed_aid": int(proposal[5]["seed_aid"]),
            "reference_aid": proposal[3],
            "scope": str(proposal[5]["scope"]),
            "target_law_id": str(proposal[5].get("target_law_id", "")),
            "base_rank": proposal[0],
        }
        for proposal in chosen
    ]
    if not expansions:
        return list(selected), records
    return [*selected[: top_k - len(expansions)], *expansions], records


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
        chunk_candidate_selection: str = "top_chunks",
        neighbor_expansion_enabled: bool = False,
        neighbor_seed_top_k: int = 5,
        neighbor_radius: int = 1,
        neighbor_max_expansions: int = 4,
        reference_adjacency: Mapping[int, Sequence[Mapping[str, object]]] | None = None,
        reference_graph_path: str | None = None,
        reference_scopes: Sequence[str] = ("same_document",),
        reference_seed_top_k: int = 5,
        reference_max_expansions: int = 4,
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
        self.chunk_candidate_selection = _validate_chunk_candidate_selection(
            chunk_candidate_selection
        )
        if neighbor_seed_top_k <= 0 or neighbor_radius <= 0 or neighbor_max_expansions <= 0:
            raise ValueError("Same-law neighbor expansion parameters must be positive")
        self.neighbor_expansion_enabled = neighbor_expansion_enabled
        self.neighbor_seed_top_k = neighbor_seed_top_k
        self.neighbor_radius = neighbor_radius
        self.neighbor_max_expansions = neighbor_max_expansions
        self._neighbor_aids: dict[int, list[tuple[int, int]]] = {}
        self._neighbor_expansion_records: list[dict[str, int]] = []
        if reference_seed_top_k <= 0 or reference_max_expansions <= 0:
            raise ValueError("Reference expansion parameters must be positive")
        self.reference_adjacency = dict(reference_adjacency or {})
        self.reference_expansion_enabled = reference_adjacency is not None
        self.reference_graph_path = reference_graph_path
        self.reference_scopes = tuple(reference_scopes)
        self.reference_seed_top_k = reference_seed_top_k
        self.reference_max_expansions = reference_max_expansions
        self._reference_expansion_records: list[dict[str, object]] = []
        self._component_results: dict[str, list[RetrievalResult]] = {}

    def fit(self, articles: Sequence[Article]) -> None:
        self._neighbor_aids = _same_law_neighbors(articles, radius=self.neighbor_radius)
        started = time.perf_counter()
        print(
            f"[hybrid] Sparse fit start articles={len(articles):,}",
            flush=True,
        )
        cuda_snapshot("retrieval_fit:sparse:before")
        self.sparse.fit(articles)
        cuda_snapshot("retrieval_fit:sparse:after")
        print(
            f"[hybrid] Sparse fit complete seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        started = time.perf_counter()
        print(
            f"[hybrid] Dense fit start articles={len(articles):,}",
            flush=True,
        )
        cuda_snapshot("retrieval_fit:dense:before")
        self.dense.fit(articles)
        cuda_snapshot("retrieval_fit:dense:after")
        print(
            f"[hybrid] Dense fit complete seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )

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
        started = time.perf_counter()
        print(
            f"[hybrid] Sparse query retrieval start queries={len(queries):,} "
            f"top_k={self.sparse_top_k}",
            flush=True,
        )
        sparse_results = self.sparse.retrieve(queries, top_k=self.sparse_top_k)
        cuda_snapshot("retrieval_query:sparse:after")
        print(
            "[hybrid] Sparse query retrieval complete "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        started = time.perf_counter()
        print(
            f"[hybrid] Dense query retrieval start queries={len(queries):,} "
            f"top_k={self.dense_top_k}",
            flush=True,
        )
        dense_results = self.dense.retrieve(queries, top_k=self.dense_top_k)
        cuda_snapshot("retrieval_query:dense:after")
        print(
            "[hybrid] Dense query retrieval complete "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
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
        if self._neighbor_expansion_records:
            records["same_law_neighbors"] = list(self._neighbor_expansion_records)
        if self._reference_expansion_records:
            records["explicit_references"] = list(self._reference_expansion_records)
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
        self._neighbor_expansion_records = []
        self._reference_expansion_records = []
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
            selected = _select_chunk_candidates(
                ranked,
                top_k=top_k,
                selection=self.chunk_candidate_selection,
            )
            if self.neighbor_expansion_enabled:
                selected, records = _expand_with_available_neighbors(
                    ranked,
                    selected,
                    qid=result.qid,
                    top_k=top_k,
                    neighbors=self._neighbor_aids,
                    seed_top_k=self.neighbor_seed_top_k,
                    max_expansions=self.neighbor_max_expansions,
                )
                self._neighbor_expansion_records.extend(records)
            if self.reference_expansion_enabled:
                selected, records = _expand_with_available_references(
                    ranked,
                    selected,
                    qid=result.qid,
                    top_k=top_k,
                    references=self.reference_adjacency,
                    seed_top_k=self.reference_seed_top_k,
                    max_expansions=self.reference_max_expansions,
                )
                self._reference_expansion_records.extend(records)
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
            "chunk_candidate_selection": self.chunk_candidate_selection,
            "same_law_neighbor_expansion": {
                "enabled": self.neighbor_expansion_enabled,
                "seed_top_k": self.neighbor_seed_top_k,
                "radius": self.neighbor_radius,
                "max_expansions": self.neighbor_max_expansions,
                "available_chunk_only": True,
            },
            "reference_expansion": {
                "enabled": self.reference_expansion_enabled,
                "graph_path": self.reference_graph_path,
                "scopes": list(self.reference_scopes),
                "seed_top_k": self.reference_seed_top_k,
                "max_expansions": self.reference_max_expansions,
                "available_chunk_only": True,
                "fixed_candidate_budget": True,
            },
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
        chunk_candidate_selection: str = "top_chunks",
        neighbor_expansion_enabled: bool = False,
        neighbor_seed_top_k: int = 5,
        neighbor_radius: int = 1,
        neighbor_max_expansions: int = 4,
        reference_adjacency: Mapping[int, Sequence[Mapping[str, object]]] | None = None,
        reference_graph_path: str | None = None,
        reference_scopes: Sequence[str] = ("same_document",),
        reference_seed_top_k: int = 5,
        reference_max_expansions: int = 4,
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
        self.chunk_candidate_selection = _validate_chunk_candidate_selection(
            chunk_candidate_selection
        )
        if neighbor_seed_top_k <= 0 or neighbor_radius <= 0 or neighbor_max_expansions <= 0:
            raise ValueError("Same-law neighbor expansion parameters must be positive")
        self.neighbor_expansion_enabled = neighbor_expansion_enabled
        self.neighbor_seed_top_k = neighbor_seed_top_k
        self.neighbor_radius = neighbor_radius
        self.neighbor_max_expansions = neighbor_max_expansions
        self._neighbor_aids: dict[int, list[tuple[int, int]]] = {}
        self._neighbor_expansion_records: list[dict[str, int]] = []
        if reference_seed_top_k <= 0 or reference_max_expansions <= 0:
            raise ValueError("Reference expansion parameters must be positive")
        self.reference_adjacency = dict(reference_adjacency or {})
        self.reference_expansion_enabled = reference_adjacency is not None
        self.reference_graph_path = reference_graph_path
        self.reference_scopes = tuple(reference_scopes)
        self.reference_seed_top_k = reference_seed_top_k
        self.reference_max_expansions = reference_max_expansions
        self._reference_expansion_records: list[dict[str, object]] = []
        self._component_results: dict[str, list[RetrievalResult]] = {}

    def fit(self, articles: Sequence[Article]) -> None:
        self._neighbor_aids = _same_law_neighbors(articles, radius=self.neighbor_radius)
        cuda_snapshot("retrieval_fit:sparse:before")
        self.sparse.fit(articles)
        cuda_snapshot("retrieval_fit:sparse:after")
        for name, retriever in self.dense_retrievers.items():
            cuda_snapshot(f"retrieval_fit:dense:{name}:before")
            retriever.fit(articles)
            cuda_snapshot(f"retrieval_fit:dense:{name}:after")

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        components = {"sparse": self.sparse.retrieve(queries, top_k=self.sparse_top_k)}
        cuda_snapshot("retrieval_query:sparse:after")
        for name, retriever in self.dense_retrievers.items():
            components[f"dense:{name}"] = retriever.retrieve(
                queries, top_k=self.dense_top_k[name]
            )
            cuda_snapshot(f"retrieval_query:dense:{name}:after")
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
        if self._neighbor_expansion_records:
            records["same_law_neighbors"] = list(self._neighbor_expansion_records)
        if self._reference_expansion_records:
            records["explicit_references"] = list(self._reference_expansion_records)
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
        self._neighbor_expansion_records = []
        self._reference_expansion_records = []
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
            )
            selected = _select_chunk_candidates(
                ranked,
                top_k=top_k,
                selection=self.chunk_candidate_selection,
            )
            if self.neighbor_expansion_enabled:
                selected, records = _expand_with_available_neighbors(
                    ranked,
                    selected,
                    qid=query.qid,
                    top_k=top_k,
                    neighbors=self._neighbor_aids,
                    seed_top_k=self.neighbor_seed_top_k,
                    max_expansions=self.neighbor_max_expansions,
                )
                self._neighbor_expansion_records.extend(records)
            if self.reference_expansion_enabled:
                selected, records = _expand_with_available_references(
                    ranked,
                    selected,
                    qid=query.qid,
                    top_k=top_k,
                    references=self.reference_adjacency,
                    seed_top_k=self.reference_seed_top_k,
                    max_expansions=self.reference_max_expansions,
                )
                self._reference_expansion_records.extend(records)
            fused.append(
                ChunkRetrievalResult(
                    qid=query.qid,
                    chunks=tuple(item[0] for item in selected),
                    scores=tuple(float(item[1]) for item in selected),
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
            "chunk_candidate_selection": self.chunk_candidate_selection,
            "same_law_neighbor_expansion": {
                "enabled": self.neighbor_expansion_enabled,
                "seed_top_k": self.neighbor_seed_top_k,
                "radius": self.neighbor_radius,
                "max_expansions": self.neighbor_max_expansions,
                "available_chunk_only": True,
            },
            "reference_expansion": {
                "enabled": self.reference_expansion_enabled,
                "graph_path": self.reference_graph_path,
                "scopes": list(self.reference_scopes),
                "seed_top_k": self.reference_seed_top_k,
                "max_expansions": self.reference_max_expansions,
                "available_chunk_only": True,
                "fixed_candidate_budget": True,
            },
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
