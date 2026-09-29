import unittest

from drill_ir.data import Article
from drill_ir.structure import audit_clause_parser


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


if __name__ == "__main__":
    unittest.main()
