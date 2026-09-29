import tempfile
import unittest
from pathlib import Path

import numpy as np

from drill_ir.data import Article, Query
from drill_ir.preprocessing import Chunk
from drill_ir.retrieval import DenseRetriever, EmbeddingCache, ExactVectorSearch


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]], *, cache_key: str = "fake-v1") -> None:
        self.vectors = vectors
        self._cache_key = cache_key
        self.document_calls = 0
        self.query_calls = 0

    @property
    def cache_key(self) -> str:
        return self._cache_key

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls += 1
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self.query_calls += 1
        return self._encode(texts)


class DenseRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.articles = [
            Article(aid=20, law_id="L1", text="finance"),
            Article(aid=10, law_id="L2", text="school"),
            Article(aid=30, law_id="L3", text="mixed"),
        ]
        self.vectors = {
            "finance": [1.0, 0.0],
            "school": [0.0, 1.0],
            "mixed": [1.0, 1.0],
            "money question": [1.0, 0.0],
        }

    def test_cosine_retrieval_uses_common_result_interface(self) -> None:
        retriever = DenseRetriever(
            encoder=FakeEncoder(self.vectors),
            search=ExactVectorSearch(metric="cosine", query_batch_size=1),
        )
        retriever.fit(self.articles)
        results = retriever.retrieve(
            [Query(qid=7, question="money question", relevant_aids=(20,))], top_k=3
        )
        self.assertEqual(results[0].qid, 7)
        self.assertEqual(results[0].aids, (20, 30, 10))
        self.assertGreater(results[0].scores[0], results[0].scores[1])

    def test_equal_scores_are_broken_by_ascending_aid(self) -> None:
        search = ExactVectorSearch(metric="dot_product")
        indices, _ = search.search(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            [20, 10, 30],
            top_k=2,
        )
        self.assertEqual(indices.tolist(), [[1, 0]])

    def test_valid_cache_avoids_reencoding_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(Path(directory))
            first_encoder = FakeEncoder(self.vectors)
            first = DenseRetriever(
                encoder=first_encoder,
                search=ExactVectorSearch(),
                cache=cache,
            )
            first.fit(self.articles)
            self.assertEqual(first_encoder.document_calls, 1)

            second_encoder = FakeEncoder(self.vectors)
            second = DenseRetriever(
                encoder=second_encoder,
                search=ExactVectorSearch(),
                cache=cache,
            )
            second.fit(self.articles)
            self.assertEqual(second_encoder.document_calls, 0)

    def test_rejects_zero_vector_for_cosine_search(self) -> None:
        with self.assertRaises(ValueError):
            ExactVectorSearch(metric="cosine").search(
                np.asarray([[0.0, 0.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                [1],
                top_k=1,
            )

    def test_chunk_scores_are_max_aggregated_to_unique_aids(self) -> None:
        class FakeChunker:
            cache_key = "fake-chunks-v1"

            def chunk_articles(self, articles: list[Article]) -> list[Chunk]:
                del articles
                return [
                    Chunk("10:0", 10, "L1", 0, "test", "weak", 0, 4, 1),
                    Chunk("10:1", 10, "L1", 1, "test", "strong", 5, 11, 1),
                    Chunk("20:0", 20, "L2", 0, "test", "medium", 0, 6, 1),
                ]

        vectors = {
            "weak": [0.0, 1.0],
            "strong": [1.0, 0.0],
            "medium": [0.8, 0.2],
            "query": [1.0, 0.0],
        }
        retriever = DenseRetriever(
            encoder=FakeEncoder(vectors),
            search=ExactVectorSearch(metric="cosine"),
            chunker=FakeChunker(),
        )
        retriever.fit(self.articles)
        result = retriever.retrieve([Query(1, "query", ())], top_k=2)[0]
        self.assertEqual(result.aids, (10, 20))
        self.assertEqual(len(result.aids), len(set(result.aids)))


if __name__ == "__main__":
    unittest.main()
