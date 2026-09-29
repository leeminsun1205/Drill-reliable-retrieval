import unittest

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

    def test_configurable_normalization_and_filters(self) -> None:
        processor = build_token_processor(
            {
                "normalization": {"strip_accents": True},
                "tokenizer": {
                    "backend": "regex_words",
                    "keep_numbers": False,
                    "stopwords": ["va"],
                    "min_token_length": 2,
                },
            }
        )
        self.assertEqual(processor("ĐIỀU 12 và án lệ"), ["dieu", "an", "le"])

    def test_character_ngrams_are_explicit(self) -> None:
        processor = build_token_processor(
            {
                "normalization": {"lowercase": False},
                "tokenizer": {
                    "backend": "character_ngram",
                    "character_ngram_min": 2,
                    "character_ngram_max": 2,
                },
            }
        )
        self.assertEqual(processor("ABC"), ["AB", "BC"])

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


if __name__ == "__main__":
    unittest.main()
