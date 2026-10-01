import unittest
import random

import numpy as np

from drill_ir.data import Article, Query, RetrievalResult
from drill_ir.preprocessing import Chunk
from drill_ir.reranking import (
    CrossEncoderReranker,
    prefilter_representations,
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

    def score(self, pairs):
        self.seen_pairs.extend(pairs)
        return np.asarray([self.scores[pair] for pair in pairs], dtype=np.float32)

    def metadata(self):
        return {"fake": True}


class CrossEncoderRerankerTest(unittest.TestCase):
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
