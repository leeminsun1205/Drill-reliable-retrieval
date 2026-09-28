import unittest

from drill_ir.data import Article, Query
from drill_ir.retrieval import BM25Retriever, tokenize


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


if __name__ == "__main__":
    unittest.main()
