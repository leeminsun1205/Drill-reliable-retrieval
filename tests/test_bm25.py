import unittest
import math

from drill_ir.data import Article, Query
from drill_ir.retrieval import BM25Retriever, tokenize
from drill_ir.preprocessing import build_token_processor


class BM25Test(unittest.TestCase):
    def test_unicode_tokenization_is_normalized_and_lowercase(self) -> None:
        self.assertEqual(tokenize("KẾT hôn!"), ["kết", "hôn"])

    def test_retrieves_matching_article_first(self) -> None:
        articles = [
            Article(10, "law-a", "quy định đăng ký kết hôn"),
            Article(20, "law-b", "quy định hoạt động ngân hàng"),
        ]
        query = Query(1, "đăng ký kết hôn", ())
        retriever = BM25Retriever()
        retriever.fit(articles)
        result = retriever.retrieve([query], top_k=2)[0]
        self.assertEqual(result.aids[0], 10)
        self.assertEqual(len(result.aids), 2)

    def test_bm25_plus_adds_delta_only_to_matching_documents(self) -> None:
        articles = [
            Article(10, "law-a", "alpha alpha"),
            Article(20, "law-b", "alpha beta beta beta"),
            Article(30, "law-c", "gamma"),
        ]
        query = Query(1, "alpha", ())
        bm25 = BM25Retriever(variant="bm25", delta=0.75)
        bm25_plus = BM25Retriever(variant="bm25_plus", delta=0.75)
        bm25.fit(articles)
        bm25_plus.fit(articles)
        base = bm25.retrieve([query], top_k=3)[0]
        plus = bm25_plus.retrieve([query], top_k=3)[0]
        base_scores = dict(zip(base.aids, base.scores))
        plus_scores = dict(zip(plus.aids, plus.scores))
        idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))
        for aid in (10, 20):
            self.assertAlmostEqual(
                plus_scores[aid] - base_scores[aid], 0.75 * idf
            )
        self.assertEqual(base_scores[30], 0.0)
        self.assertEqual(plus_scores[30], 0.0)

    def test_bm25_plus_configuration_is_reported(self) -> None:
        retriever = BM25Retriever(variant="bm25_plus", delta=0.5)
        retriever.fit([Article(10, "law", "alpha")])
        metadata = retriever.run_metadata()
        self.assertEqual(metadata["variant"], "bm25_plus")
        self.assertEqual(metadata["delta"], 0.5)

    def test_rejects_unknown_bm25_variant_and_negative_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "variant"):
            BM25Retriever(variant="unknown")
        with self.assertRaisesRegex(ValueError, "delta"):
            BM25Retriever(variant="bm25_plus", delta=-0.1)

    def test_vncorenlp_requires_model_path(self) -> None:
        processor = build_token_processor(
            {"tokenizer": {"word_segmenter": "vncorenlp"}}
        )
        with self.assertRaisesRegex(ValueError, "vncorenlp_jar_path"):
            processor("đăng ký kết hôn")

    def test_vncorenlp_bounds_internal_requests(self) -> None:
        class FakeSegmenter:
            def __init__(self) -> None:
                self.requests: list[str] = []

            def tokenize(self, text: str) -> list[list[str]]:
                self.requests.append(text)
                return [text.split()]

        processor = build_token_processor(
            {
                "tokenizer": {
                    "word_segmenter": "vncorenlp",
                    "vncorenlp_jar_path": "unused-in-test.jar",
                    "segmenter_max_characters": 12,
                }
            }
        )
        fake = FakeSegmenter()
        processor._segmenter = fake
        self.assertEqual(
            processor("mot hai ba bon nam sau bay"),
            ["mot", "hai", "ba", "bon", "nam", "sau", "bay"],
        )
        self.assertGreater(len(fake.requests), 1)
        self.assertTrue(all(len(request) <= 12 for request in fake.requests))

    def test_stopword_presets_are_selectable(self) -> None:
        conservative = build_token_processor(
            {"tokenizer": {"stopword_preset": "vi_legal_conservative"}}
        )
        self.assertEqual(conservative("người phải đăng ký và nộp thuế"), ["người", "phải", "đăng", "ký", "nộp", "thuế"])

    def test_document_frequency_pruning(self) -> None:
        articles = [
            Article(10, "law-a", "chung alpha"),
            Article(20, "law-b", "chung beta"),
            Article(30, "law-c", "chung beta"),
        ]
        retriever = BM25Retriever(
            min_document_frequency=2,
            max_document_frequency_ratio=0.9,
        )
        retriever.fit(articles)
        metadata = retriever.run_metadata()["vocabulary"]
        self.assertEqual(metadata["original_terms"], 3)
        self.assertEqual(metadata["retained_terms"], 1)
        self.assertEqual(metadata["pruned_terms"], 2)
        result = retriever.retrieve([Query(1, "beta", ())], top_k=1)[0]
        self.assertEqual(result.aids[0], 20)


if __name__ == "__main__":
    unittest.main()
