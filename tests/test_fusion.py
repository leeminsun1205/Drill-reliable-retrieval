import unittest

from drill_ir.data import Article, Query, RetrievalResult
from drill_ir.retrieval import WeightedRRFRetriever


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


class WeightedRRFTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
