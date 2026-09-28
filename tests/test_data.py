import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from drill_ir.data import iter_articles, load_queries, validate_dataset


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.corpus = root / "corpus.parquet"
        self.queries = root / "queries.parquet"
        corpus_schema = pa.schema(
            [
                ("id", pa.int64()),
                ("law_id", pa.string()),
                (
                    "content",
                    pa.list_(pa.struct([("aid", pa.int64()), ("content_Article", pa.string())])),
                ),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "id": 0,
                        "law_id": "law-a",
                        "content": [
                            {"aid": 10, "content_Article": "Nội dung"},
                            {"aid": 11, "content_Article": " "},
                        ],
                    }
                ],
                schema=corpus_schema,
            ),
            self.corpus,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [{"qid": 7, "question": "Câu hỏi?", "relevant_laws": [10, 10]}]
            ),
            self.queries,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_blank_articles_are_reported_and_excluded_by_default(self) -> None:
        report = validate_dataset(self.corpus, self.queries)
        self.assertEqual(report.blank_articles, 1)
        self.assertEqual(report.duplicate_gold_rows, 1)
        self.assertEqual([article.aid for article in iter_articles(self.corpus)], [10])

    def test_duplicate_gold_is_deduplicated_stably(self) -> None:
        loaded = load_queries(self.queries)
        self.assertEqual(loaded[0].qid, 7)
        self.assertEqual(loaded[0].relevant_aids, (10,))


if __name__ == "__main__":
    unittest.main()
