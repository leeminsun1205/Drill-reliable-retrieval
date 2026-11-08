import unittest

from drill_ir.analysis.error_audit import audit_gold_flow


class ErrorAuditTests(unittest.TestCase):
    def setUp(self):
        self.gold = {1: [10, 20, 30, 40, 50], 2: [70]}
        self.pool = {1: [20, 30, 40, 50, 60], 2: [70, 80]}
        self.reranker = {1: [40, 50, 30, 20, 60], 2: [70, 80]}
        self.lgbm = {1: [40, 20, 30, 50, 60], 2: [70, 80]}
        self.selected = {1: [40], 2: [70, 80]}

    def test_each_gold_assignment_is_counted_once(self):
        summary, rows = audit_gold_flow(
            self.gold, self.pool, self.reranker, self.lgbm, self.selected,
            maximum_k=2,
        )
        self.assertEqual(summary["gold_assignments"], 6)
        self.assertEqual(summary["gold_partition"], {
            "outside_pool": 1, "cut_by_prefix": 1,
            "low_in_both_rankings": 1, "recovered": 2, "demoted_by_lgbm": 1,
        })
        self.assertEqual(summary["query_outcomes"], {"missing_only": 1, "extra_only": 1})
        self.assertEqual(summary["false_negative_assignments"], 4)
        self.assertEqual(summary["false_positive_assignments"], 1)
        # Per-query precision 1 and 1/2; recall 1/5 and 1.
        self.assertAlmostEqual(summary["final_metrics"]["precision"], 0.75)
        self.assertAlmostEqual(summary["final_metrics"]["recall"], 0.6)
        self.assertAlmostEqual(summary["final_metrics"]["f2"], 0.625)
        self.assertEqual(rows[0]["missing"], [10, 20, 30, 50])

    def test_inconsistent_candidate_sets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Candidate sets differ"):
            audit_gold_flow(self.gold, {1: [40], 2: [70, 80]}, self.reranker, self.lgbm, self.selected)

    def test_nonprefix_predictions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a nonempty prefix"):
            audit_gold_flow(self.gold, self.pool, self.reranker, self.lgbm, {1: [20], 2: [70, 80]})


if __name__ == "__main__":
    unittest.main()
