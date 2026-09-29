import unittest

import numpy as np

from drill_ir.data import Article, Query, RetrievalResult
from drill_ir.preprocessing import Chunk
from drill_ir.reranking import CrossEncoderReranker


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


if __name__ == "__main__":
    unittest.main()
