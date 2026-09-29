import unittest

from drill_ir.data import Article
from drill_ir.structure import audit_clause_parser, sample_articles_by_law


class ClauseAuditTest(unittest.TestCase):
    def test_audit_is_deterministic_and_stratified_by_status(self) -> None:
        articles = [
            Article(1, "a", "1. Một\n2. Hai"),
            Article(2, "b", "không cấu trúc"),
            Article(3, "c", "1. Một\n3. Ba"),
        ]
        first = audit_clause_parser(articles, seed=42, samples_per_status=1)
        second = audit_clause_parser(articles, seed=42, samples_per_status=1)
        self.assertEqual(first, second)
        summary, records = first
        self.assertEqual(summary["status_counts"], {
            "ambiguous_sequence": 1,
            "no_structure": 1,
            "parsed": 1,
        })
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["review"]["boundary_correct"] is None for record in records))

    def test_law_sampling_keeps_every_article_for_selected_laws(self) -> None:
        articles = [
            Article(1, "a", "one"),
            Article(2, "a", "two"),
            Article(3, "b", "three"),
            Article(4, "c", "four"),
        ]
        selected, laws = sample_articles_by_law(articles, law_count=2, seed=42)
        self.assertEqual({article.law_id for article in selected}, set(laws))
        for law_id in laws:
            self.assertEqual(
                [article.aid for article in selected if article.law_id == law_id],
                [article.aid for article in articles if article.law_id == law_id],
            )

    def test_law_sampling_excludes_prior_laws(self) -> None:
        articles = [
            Article(1, "a", "one"),
            Article(2, "b", "two"),
            Article(3, "c", "three"),
        ]
        selected, laws = sample_articles_by_law(
            articles,
            law_count=2,
            seed=42,
            excluded_law_ids=["b"],
        )
        self.assertEqual(laws, ["a", "c"])
        self.assertEqual([article.aid for article in selected], [1, 3])


if __name__ == "__main__":
    unittest.main()
