import unittest
import math
from unittest.mock import patch

from drill_ir.data import Article, Query, RetrievalResult
from drill_ir.preprocessing import Chunk
from drill_ir.retrieval import (
    ChunkRetrievalResult,
    MultiDenseWeightedRRFRetriever,
    WeightedRRFRetriever,
    build_retriever,
)


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.fit_calls = 0
        self.requested_top_k = None

    def fit(self, articles) -> None:
        self.fit_calls += 1

    def retrieve(self, queries, top_k: int) -> list[RetrievalResult]:
        self.requested_top_k = top_k
        return [
            RetrievalResult(result.qid, result.aids[:top_k], result.scores[:top_k])
            for result in self.results
        ]


class FakeChunkRetriever(FakeRetriever):
    def __init__(self, results, chunk_results):
        super().__init__(results)
        self.chunk_results = chunk_results

    def retrieve_chunks(self, queries, top_k):
        return [
            ChunkRetrievalResult(result.qid, result.chunks[:top_k], result.scores[:top_k])
            for result in self.chunk_results
        ]


class WeightedRRFTest(unittest.TestCase):
    def test_weighted_rrf_matches_standard_one_based_formula(self) -> None:
        sparse = FakeRetriever([RetrievalResult(1, (10, 20), (9.0, 8.0))])
        dense = FakeRetriever([RetrievalResult(1, (20, 30), (0.9, 0.8))])
        retriever = WeightedRRFRetriever(
            sparse=sparse,
            dense=dense,
            sparse_weight=0.25,
            dense_weight=0.75,
            rrf_k=60,
            sparse_top_k=2,
            dense_top_k=2,
        )
        result = retriever.retrieve([Query(1, "q", ())], top_k=3)[0]
        scores = dict(zip(result.aids, result.scores))
        self.assertTrue(math.isclose(scores[10], 0.25 / 61))
        self.assertTrue(math.isclose(scores[20], 0.25 / 62 + 0.75 / 61))
        self.assertTrue(math.isclose(scores[30], 0.75 / 62))

    def test_fuses_ranks_and_preserves_component_depths(self) -> None:
        sparse = FakeRetriever([RetrievalResult(1, (10, 20), (9.0, 8.0))])
        dense = FakeRetriever([RetrievalResult(1, (20, 30), (0.9, 0.8))])
        retriever = WeightedRRFRetriever(
            sparse=sparse,
            dense=dense,
            rrf_k=60,
            sparse_top_k=2,
            dense_top_k=2,
        )
        retriever.fit([Article(10, "law", "text")])
        result = retriever.retrieve([Query(1, "q", ())], top_k=3)[0]
        self.assertEqual(result.aids, (20, 10, 30))
        self.assertEqual(sparse.fit_calls, 1)
        self.assertEqual(dense.fit_calls, 1)
        self.assertEqual(sparse.requested_top_k, 2)
        self.assertEqual(dense.requested_top_k, 2)
        self.assertGreater(result.scores[0], result.scores[1])

    def test_weights_and_aid_tie_break_are_deterministic(self) -> None:
        sparse = FakeRetriever([RetrievalResult(1, (20,), (100.0,))])
        dense = FakeRetriever([RetrievalResult(1, (10,), (-100.0,))])
        retriever = WeightedRRFRetriever(sparse=sparse, dense=dense, rrf_k=60)
        result = retriever.retrieve([Query(1, "q", ())], top_k=2)[0]
        self.assertEqual(result.aids, (10, 20))
        self.assertEqual(result.scores[0], result.scores[1])

    def test_rejects_invalid_weights_and_missing_queries(self) -> None:
        fake = FakeRetriever([])
        with self.assertRaisesRegex(ValueError, "weight"):
            WeightedRRFRetriever(sparse=fake, dense=fake, sparse_weight=0, dense_weight=0)
        retriever = WeightedRRFRetriever(sparse=fake, dense=fake)
        with self.assertRaisesRegex(ValueError, "every query"):
            retriever.retrieve([Query(1, "q", ())], top_k=1)

    def test_ignores_sparse_zero_score_fillers(self) -> None:
        sparse = FakeRetriever([RetrievalResult(1, (10, 20), (1.0, 0.0))])
        dense = FakeRetriever([RetrievalResult(1, (30,), (0.5,))])
        retriever = WeightedRRFRetriever(sparse=sparse, dense=dense)
        result = retriever.retrieve([Query(1, "q", ())], top_k=3)[0]
        self.assertEqual(result.aids, (10, 30))

    def test_fuses_sparse_article_rank_into_dense_chunk_candidates(self) -> None:
        chunks = (
            Chunk("20:0", 20, "L", 0, "diem", "a", 0, 1, 1),
            Chunk("10:0", 10, "L", 0, "diem", "b", 0, 1, 1),
            Chunk("20:1", 20, "L", 1, "diem", "c", 0, 1, 1),
        )
        sparse = FakeRetriever([RetrievalResult(1, (10, 20), (2.0, 1.0))])
        dense = FakeChunkRetriever(
            [RetrievalResult(1, (20, 10), (1.0, 0.5))],
            [ChunkRetrievalResult(1, chunks, (0.9, 0.8, 0.7))],
        )
        retriever = WeightedRRFRetriever(
            sparse=sparse, dense=dense, sparse_weight=0.8, dense_weight=0.2, rrf_k=20
        )
        retriever.retrieve([Query(1, "q", ())], top_k=2)
        result = retriever.retrieve_chunks(
            [Query(1, "q", ())], top_k=2, dense_depth=3
        )[0]
        self.assertEqual(tuple(chunk.aid for chunk in result.chunks), (10, 20))

    def test_multi_dense_fuses_articles_and_exact_matching_chunks(self) -> None:
        shared = Chunk("20:a", 20, "L", 0, "diem", "shared", 0, 6, 1)
        other = Chunk("30:a", 30, "L", 0, "diem", "other", 0, 5, 1)
        sparse = FakeRetriever([RetrievalResult(1, (10,), (2.0,))])
        dense_a = FakeChunkRetriever(
            [RetrievalResult(1, (20, 30), (0.9, 0.8))],
            [ChunkRetrievalResult(1, (shared, other), (0.9, 0.8))],
        )
        dense_b = FakeChunkRetriever(
            [RetrievalResult(1, (20, 10), (0.7, 0.6))],
            [ChunkRetrievalResult(1, (shared,), (0.7,))],
        )
        retriever = MultiDenseWeightedRRFRetriever(
            sparse=sparse,
            dense_retrievers={"a": dense_a, "b": dense_b},
            sparse_weight=0.2,
            dense_weights={"a": 0.4, "b": 0.4},
            rrf_k=20,
            sparse_top_k=2,
            dense_top_k={"a": 2, "b": 2},
        )
        result = retriever.retrieve([Query(1, "q", ())], top_k=3)[0]
        self.assertEqual(result.aids[0], 20)
        self.assertEqual(set(retriever.component_results()), {"sparse", "dense:a", "dense:b"})
        chunks = retriever.retrieve_chunks(
            [Query(1, "q", ())], top_k=1, dense_depth=2
        )[0]
        self.assertEqual(chunks.chunks, (shared,))

    def test_factory_builds_named_dense_profiles_without_mutating_base(self) -> None:
        base_dense = {
            "encoder": {
                "backend": "sentence_transformers",
                "model_name": "base",
                "max_seq_length": 512,
            },
            "search": {"backend": "exact", "metric": "cosine"},
            "chunking": {"method": "article_truncate", "aggregation": "max"},
            "cache": {"enabled": False},
        }
        config = {
            "hybrid": {
                "fusion": "weighted_rrf",
                "sparse_weight": 0.2,
                "dense_components": [
                    {"name": "first", "profile": "a", "weight": 0.4},
                    {"name": "second", "profile": "b", "weight": 0.4},
                ],
            },
            "bm25": {},
            "dense": base_dense,
            "dense_profiles": {
                "a": {"encoder": {"model_name": "model-a"}},
                "b": {"encoder": {"model_name": "model-b"}},
            },
        }
        with patch("drill_ir.retrieval.factory.SentenceTransformerEncoder") as encoder:
            retriever = build_retriever("hybrid", config, {"accelerator": "cpu"})
        self.assertIsInstance(retriever, MultiDenseWeightedRRFRetriever)
        self.assertEqual(set(retriever.dense_retrievers), {"first", "second"})
        self.assertEqual(
            [call.args[0].model_name for call in encoder.call_args_list],
            ["model-a", "model-b"],
        )
        self.assertEqual(base_dense["encoder"]["model_name"], "base")


if __name__ == "__main__":
    unittest.main()
