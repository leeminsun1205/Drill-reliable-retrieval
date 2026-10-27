import tempfile
import unittest
from pathlib import Path

import numpy as np

from drill_ir.data import Article, Query
from drill_ir.preprocessing import Chunk
from drill_ir.retrieval import (
    DenseChunkCache,
    DenseRetriever,
    EmbeddingCache,
    ExactVectorSearch,
    FaissFlatVectorSearch,
)


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]], *, cache_key: str = "fake-v1") -> None:
        self.vectors = vectors
        self._cache_key = cache_key
        self.document_calls = 0
        self.query_calls = 0
        self.closed = False

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

    def close(self) -> None:
        self.closed = True


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

    def test_faiss_flat_matches_numpy_exact_chunk_and_grouped_results(self) -> None:
        rng = np.random.default_rng(42)
        documents = rng.normal(size=(257, 16)).astype(np.float32)
        queries = rng.normal(size=(7, 16)).astype(np.float32)
        aids = np.asarray([1000 + index // 3 for index in range(len(documents))])
        numpy_search = ExactVectorSearch(metric="cosine", query_batch_size=3)
        faiss_search = FaissFlatVectorSearch(
            metric="cosine", query_batch_size=3, device="cpu"
        )

        numpy_indices, numpy_scores = numpy_search.search(
            queries, documents, aids, top_k=25
        )
        faiss_indices, faiss_scores = faiss_search.search(
            queries, documents, aids, top_k=25
        )
        self.assertEqual(faiss_indices.tolist(), numpy_indices.tolist())
        np.testing.assert_allclose(faiss_scores, numpy_scores, rtol=1e-5, atol=1e-6)

        numpy_aids, numpy_grouped_scores = numpy_search.search_grouped(
            queries, documents, aids, top_k=25
        )
        faiss_aids, faiss_grouped_scores = faiss_search.search_grouped(
            queries, documents, aids, top_k=25
        )
        self.assertEqual(faiss_aids.tolist(), numpy_aids.tolist())
        np.testing.assert_allclose(
            faiss_grouped_scores, numpy_grouped_scores, rtol=1e-5, atol=1e-6
        )

    def test_faiss_flat_expands_through_duplicate_aids_and_score_ties(self) -> None:
        documents = np.asarray(
            [[1.0, 0.0]] * 80 + [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
            dtype=np.float32,
        )
        aids = np.asarray([20] * 80 + [30, 10, 40])
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        numpy_search = ExactVectorSearch(metric="dot_product")
        faiss_search = FaissFlatVectorSearch(metric="dot_product", device="cpu")
        expected_aids, expected_scores = numpy_search.search_grouped(
            query, documents, aids, top_k=4
        )
        actual_aids, actual_scores = faiss_search.search_grouped(
            query, documents, aids, top_k=4
        )
        self.assertEqual(actual_aids.tolist(), expected_aids.tolist())
        np.testing.assert_allclose(actual_scores, expected_scores)

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
            self.assertFalse(second._document_embeddings.flags.writeable)

    def test_chunk_cache_round_trip_preserves_full_records_and_checks_checksum(self) -> None:
        chunk = Chunk(
            "10:khoan:0", 10, "L1", 0, "khoan_diem", "Điều 1", 2, 9, 3,
            parse_status="parsed", structure_path=("khoan:1",), source_spans=((2, 9),),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = DenseChunkCache(Path(directory))
            cache.save(
                [chunk], corpus_fingerprint="corpus", chunker_key="chunker",
                chunking_stats={"method": "khoan_diem"},
            )
            loaded = cache.load(corpus_fingerprint="corpus", chunker_key="chunker")
            self.assertEqual(loaded, ([chunk], {"method": "khoan_diem"}))
            chunks_path = next(Path(directory).rglob("chunks.jsonl"))
            chunks_path.write_text(chunks_path.read_text() + "\n", encoding="utf-8")
            self.assertIsNone(
                cache.load(corpus_fingerprint="corpus", chunker_key="chunker")
            )

    def test_required_chunk_cache_rejects_miss_before_chunking(self) -> None:
        class FailingChunker:
            cache_key = "unused"

            def chunk_articles(self, articles):
                raise AssertionError("chunker must not run on a required cache miss")

        with tempfile.TemporaryDirectory() as directory:
            retriever = DenseRetriever(
                encoder=FakeEncoder(self.vectors), search=ExactVectorSearch(),
                chunker=FailingChunker(), chunk_cache=DenseChunkCache(directory),
            )
            with self.assertRaisesRegex(RuntimeError, "Required dense chunk"):
                retriever.prepare_chunks(self.articles, require_existing=True)

    def test_close_releases_dense_state_and_encoder(self) -> None:
        encoder = FakeEncoder(self.vectors)
        retriever = DenseRetriever(encoder=encoder, search=ExactVectorSearch())
        retriever.fit(self.articles)
        retriever.retrieve(
            [Query(qid=7, question="money question", relevant_aids=(20,))], top_k=2
        )
        retriever.close()
        self.assertTrue(encoder.closed)
        self.assertIsNone(retriever._document_embeddings)
        self.assertIsNone(retriever._cached_query_embeddings)
        self.assertEqual(retriever._chunks, [])
        self.assertEqual(retriever._aids, [])

    def test_required_cache_rejects_miss_without_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder(self.vectors)
            retriever = DenseRetriever(
                encoder=encoder,
                search=ExactVectorSearch(),
                cache=EmbeddingCache(Path(directory)),
                require_cache_hit=True,
            )
            with self.assertRaisesRegex(RuntimeError, "Required dense"):
                retriever.fit(self.articles)
            self.assertEqual(encoder.document_calls, 0)

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

    def test_rocchio_reuses_adapted_query_and_records_unique_feedback_aids(self) -> None:
        articles = [
            Article(10, "L1", "diagonal"),
            Article(20, "L2", "vertical"),
        ]
        vectors = {
            "diagonal": [1.0, 1.0],
            "vertical": [0.0, 1.0],
            "query": [1.0, 0.0],
        }
        encoder = FakeEncoder(vectors)
        retriever = DenseRetriever(
            encoder=encoder,
            search=ExactVectorSearch(metric="cosine"),
            rocchio_enabled=True,
            rocchio_feedback_documents=1,
            rocchio_alpha=1.0,
            rocchio_beta=1.0,
            rocchio_candidate_depth=2,
        )
        retriever.fit(articles)
        query = Query(1, "query", ())
        result = retriever.retrieve([query], top_k=2)[0]
        retriever.retrieve_chunks([query], top_k=2)
        self.assertEqual(encoder.query_calls, 1)
        self.assertGreater(result.scores[1], 0.0)
        self.assertEqual(retriever.feedback_records()[0]["feedback_aids"], [10])
        self.assertEqual(retriever.run_metadata()["feedback"]["method"], "rocchio")

    def test_rocchio_rejects_inconsistent_negative_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires gamma"):
            DenseRetriever(
                encoder=FakeEncoder(self.vectors),
                search=ExactVectorSearch(),
                rocchio_negative_documents=1,
            )


if __name__ == "__main__":
    unittest.main()
