import unittest
import random
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from drill_ir.data import Article, Query, RetrievalResult
from drill_ir.preprocessing import Chunk
from drill_ir.reranking import (
    CrossEncoderReranker,
    HFCrossEncoderConfig,
    HFCrossEncoderScorer,
    prefilter_representations,
    prepare_pairwise_training_data_from_audit,
    select_fpt_semi_hard_negatives,
    select_best_passages,
)
from drill_ir.retrieval import ChunkRetrievalResult


class FakeChunker:
    cache_key = "fake-reranker-chunks"

    def chunk_articles(self, articles):
        chunks = []
        for article in articles:
            for index, text in enumerate(article.text.split("|")):
                chunks.append(
                    Chunk(
                        chunk_id=f"{article.aid}:{index}",
                        aid=article.aid,
                        law_id=article.law_id,
                        chunk_index=index,
                        chunk_type="fake",
                        text=text,
                        char_start=0,
                        char_end=len(text),
                        token_count=1,
                    )
                )
        return chunks


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.seen_pairs = []
        self.call_sizes = []

    def score(self, pairs):
        self.call_sizes.append(len(pairs))
        self.seen_pairs.extend(pairs)
        return np.asarray([self.scores[pair] for pair in pairs], dtype=np.float32)

    def metadata(self):
        return {"fake": True}


class CrossEncoderRerankerTest(unittest.TestCase):
    def test_pairwise_training_rows_expand_audited_positive_negative_cross_product(self) -> None:
        query = Query(7, "mức phạt", (10,))
        audit_row = {
            "qid": 7,
            "positive": [
                {"aid": 10, "chunk_id": "10:0", "text": "đúng", "score": 2.0}
            ],
            "negative": [
                {"aid": 20, "chunk_id": "20:0", "text": "nhiễu 1", "score": 1.0},
                {"aid": 30, "chunk_id": "30:0", "text": "nhiễu 2", "score": 0.5},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.jsonl"
            output = root / "pairwise.jsonl"
            audit.write_text(json.dumps(audit_row, ensure_ascii=False) + "\n")
            with patch(
                "drill_ir.reranking.training.load_queries", return_value=[query]
            ):
                prepare_pairwise_training_data_from_audit(
                    audit, root / "queries.parquet", output
                )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(
            rows,
            [
                {"query": "mức phạt", "positive": "đúng", "negative": "nhiễu 1"},
                {"query": "mức phạt", "positive": "đúng", "negative": "nhiễu 2"},
            ],
        )

    def test_hf_scorer_batch_override_validates_without_loading_model(self) -> None:
        scorer = HFCrossEncoderScorer(HFCrossEncoderConfig(model_name="unused"))
        self.assertEqual(scorer.score_with_batch_size([], batch_size=16).shape, (0,))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            scorer.score_with_batch_size([], batch_size=0)

    def test_global_stream_matches_per_query_and_fills_outer_batches(self) -> None:
        articles = [
            Article(10, "law", "a|b"),
            Article(20, "law", "c"),
        ]
        queries = [Query(1, "q1", ()), Query(2, "q2", ())]
        candidates = [
            RetrievalResult(1, (10, 20), (2.0, 1.0)),
            RetrievalResult(2, (20, 10), (2.0, 1.0)),
        ]
        scores = {
            ("q1", "a"): 0.1,
            ("q1", "b"): 0.9,
            ("q1", "c"): 0.5,
            ("q2", "a"): 0.8,
            ("q2", "b"): 0.2,
            ("q2", "c"): 0.6,
        }
        per_query_scorer = FakeScorer(scores)
        per_query = CrossEncoderReranker(
            scorer=per_query_scorer,
            chunker=FakeChunker(),
            candidate_pool_size=2,
            representation="diem",
            batching="per_query",
        )
        global_scorer = FakeScorer(scores)
        global_stream = CrossEncoderReranker(
            scorer=global_scorer,
            chunker=FakeChunker(),
            candidate_pool_size=2,
            representation="diem",
            batching="global_stream",
            outer_pair_batch_size=4,
        )
        packed_scorer = FakeScorer(scores)
        packed_stream = CrossEncoderReranker(
            scorer=packed_scorer,
            chunker=FakeChunker(),
            candidate_pool_size=2,
            representation="diem",
            batching="packed_stream",
            neural_batch_size=4,
        )
        per_query.fit(articles)
        global_stream.fit(articles)
        packed_stream.fit(articles)
        expected = per_query.rerank(queries, candidates)
        actual = global_stream.rerank(queries, candidates)
        self.assertEqual(actual, expected)
        self.assertEqual(packed_stream.rerank(queries, candidates), expected)
        self.assertEqual(per_query_scorer.call_sizes, [3, 3])
        self.assertEqual(global_scorer.call_sizes, [4, 2])
        self.assertEqual(packed_scorer.call_sizes, [4, 2])
        self.assertEqual(global_stream.run_metadata()["scored_pairs"], 6)
        self.assertEqual(global_stream.run_metadata()["scorer_calls"], 2)

    def test_global_chunk_stream_matches_per_query(self) -> None:
        queries = [Query(1, "q1", ()), Query(2, "q2", ())]
        chunks_a = (
            Chunk("10:0", 10, "L", 0, "diem", "a", 0, 1, 1),
            Chunk("20:0", 20, "L", 0, "diem", "b", 0, 1, 1),
            Chunk("10:1", 10, "L", 1, "diem", "c", 1, 2, 1),
        )
        chunks_b = (
            Chunk("20:0", 20, "L", 0, "diem", "b", 0, 1, 1),
            Chunk("10:0", 10, "L", 0, "diem", "a", 0, 1, 1),
        )
        candidates = [
            ChunkRetrievalResult(1, chunks_a, (3.0, 2.0, 1.0)),
            ChunkRetrievalResult(2, chunks_b, (2.0, 1.0)),
        ]
        scores = {
            ("q1", "a"): 0.1,
            ("q1", "b"): 0.5,
            ("q1", "c"): 0.9,
            ("q2", "a"): 0.8,
            ("q2", "b"): 0.6,
        }
        controls = []
        for batching in ("per_query", "global_stream", "packed_stream"):
            reranker = CrossEncoderReranker(
                scorer=FakeScorer(scores),
                chunker=FakeChunker(),
                candidate_pool_size=3,
                representation="retrieved",
                candidate_source="retrieved_chunks",
                batching=batching,
                outer_pair_batch_size=4,
                neural_batch_size=4,
            )
            controls.append(reranker.rerank_chunks(queries, candidates))
        self.assertEqual(controls[1], controls[0])
        self.assertEqual(controls[2], controls[0])

    def test_rejects_invalid_batching_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "batching"):
            CrossEncoderReranker(
                scorer=FakeScorer({}),
                chunker=FakeChunker(),
                candidate_pool_size=1,
                representation="article",
                batching="unknown",
            )
        with self.assertRaisesRegex(ValueError, "outer_pair_batch_size"):
            CrossEncoderReranker(
                scorer=FakeScorer({}),
                chunker=FakeChunker(),
                candidate_pool_size=1,
                representation="article",
                outer_pair_batch_size=0,
            )

    def test_fpt_negatives_exclude_gold_overlength_and_top_five(self) -> None:
        selected = select_fpt_semi_hard_negatives(
            tuple(range(1, 16)),
            (2,),
            eligible_aids=set(range(1, 16)) - {4},
            top_n=12,
            skip_top=5,
            large_pool_threshold=15,
            large_sample_size=10,
            small_sample_size=5,
            rng=random.Random(42),
        )
        # Eligible non-gold ranking begins 1,3,5,6,7; these five are skipped.
        self.assertEqual(len(selected), 5)
        self.assertTrue(set(selected).issubset({8, 9, 10, 11, 12, 13, 14}))
        self.assertNotIn(2, selected)
        self.assertNotIn(4, selected)

    def test_fpt_negatives_are_deterministic_and_use_ten_for_large_pool(self) -> None:
        kwargs = dict(
            ranked_aids=tuple(range(40)),
            relevant_aids=(0,),
            eligible_aids=set(range(40)),
            rng=random.Random(7),
        )
        first = select_fpt_semi_hard_negatives(**kwargs)
        kwargs["rng"] = random.Random(7)
        second = select_fpt_semi_hard_negatives(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertTrue(set(first).isdisjoint({1, 2, 3, 4, 5}))

    def test_training_mining_selects_best_chunk_per_aid_with_stable_ties(self) -> None:
        chunks = {
            10: FakeChunker().chunk_articles([Article(10, "law", "weak|strong")]),
            20: FakeChunker().chunk_articles([Article(20, "law", "first|second")]),
        }
        selected = select_best_passages((10, 20), chunks, [0.1, 0.9, 0.5, 0.5])
        self.assertEqual(selected[10].text, "strong")
        self.assertEqual(selected[20].text, "first")

    def test_training_prefilter_caps_chunks_by_query_overlap(self) -> None:
        chunks = FakeChunker().chunk_articles(
            [Article(10, "law", "không liên quan|học văn hóa cho phạm nhân|phạm nhân")]
        )
        selected = prefilter_representations("phạm nhân học văn hóa", chunks, maximum=2)
        self.assertEqual([chunk.text for chunk in selected], ["học văn hóa cho phạm nhân", "phạm nhân"])

    def test_max_aggregates_chunks_and_limits_candidate_pool(self) -> None:
        scorer = FakeScorer(
            {
                ("query", "weak"): 0.1,
                ("query", "strong"): 0.9,
                ("query", "medium"): 0.5,
            }
        )
        reranker = CrossEncoderReranker(
            scorer=scorer,
            chunker=FakeChunker(),
            candidate_pool_size=2,
            representation="diem",
        )
        reranker.fit(
            [
                Article(10, "law", "weak|strong"),
                Article(20, "law", "medium"),
            ]
        )
        result = reranker.rerank(
            [Query(1, "query", ())],
            [RetrievalResult(1, (20, 10, 30), (3.0, 2.0, 1.0))],
        )[0]
        self.assertEqual(result.aids, (10, 20))
        self.assertEqual(result.scores, (0.8999999761581421, 0.5))
        self.assertEqual(len(scorer.seen_pairs), 3)

    def test_equal_scores_preserve_first_stage_order(self) -> None:
        scorer = FakeScorer({("query", "ten"): 1.0, ("query", "twenty"): 1.0})
        reranker = CrossEncoderReranker(
            scorer=scorer,
            chunker=FakeChunker(),
            candidate_pool_size=2,
            representation="article",
        )
        reranker.fit([Article(10, "law", "ten"), Article(20, "law", "twenty")])
        result = reranker.rerank(
            [Query(1, "query", ())],
            [RetrievalResult(1, (20, 10), (2.0, 1.0))],
        )[0]
        self.assertEqual(result.aids, (20, 10))

    def test_requires_every_candidate_representation(self) -> None:
        reranker = CrossEncoderReranker(
            scorer=FakeScorer({}),
            chunker=FakeChunker(),
            candidate_pool_size=1,
            representation="article",
        )
        reranker.fit([Article(10, "law", "ten")])
        with self.assertRaisesRegex(ValueError, "aid=20"):
            reranker.rerank(
                [Query(1, "query", ())],
                [RetrievalResult(1, (20,), (1.0,))],
            )

    def test_reranks_retrieved_chunks_then_max_aggregates_by_aid(self) -> None:
        scorer = FakeScorer(
            {("query", "weak"): 0.1, ("query", "strong"): 0.9, ("query", "other"): 0.5}
        )
        reranker = CrossEncoderReranker(
            scorer=scorer,
            chunker=FakeChunker(),
            candidate_pool_size=3,
            representation="retrieved",
            candidate_source="retrieved_chunks",
        )
        chunks = (
            Chunk("10:0", 10, "L1", 0, "diem", "weak", 0, 4, 1),
            Chunk("20:0", 20, "L2", 0, "diem", "other", 0, 5, 1),
            Chunk("10:1", 10, "L1", 1, "diem", "strong", 5, 11, 1),
        )
        result = reranker.rerank_chunks(
            [Query(1, "query", ())],
            [ChunkRetrievalResult(1, chunks, (3.0, 2.0, 1.0))],
        )[0]
        self.assertEqual(result.aids, (10, 20))
        self.assertEqual(len(scorer.seen_pairs), 3)


if __name__ == "__main__":
    unittest.main()
