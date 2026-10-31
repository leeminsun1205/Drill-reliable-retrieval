import unittest

from drill_ir.analysis import evaluate_nested_decision_policies


class DecisionPolicyTest(unittest.TestCase):
    def test_nested_policy_screen_covers_every_query(self) -> None:
        gold = {qid: ((qid * 10),) for qid in range(1, 11)}
        rankings = {
            qid: (
                [qid * 10, qid * 10 + 1, qid * 10 + 2],
                [0.9 - qid * 0.001, 0.2, 0.1],
            )
            for qid in gold
        }
        result = evaluate_nested_decision_policies(
            gold, rankings, folds=5, seed=42, fallback_k=1, maximum_k=3
        )
        self.assertEqual(
            set(result["policies"]),
            {"raw", "platt", "isotonic", "cardinality_logistic"},
        )
        for metrics in result["policies"].values():
            self.assertEqual(metrics["queries"], 10)
            self.assertEqual(len(metrics["folds"]), 5)
            self.assertEqual(metrics["f2"], 1.0)

    def test_policy_screen_rejects_mismatched_qids(self) -> None:
        with self.assertRaisesRegex(ValueError, "qids must match"):
            evaluate_nested_decision_policies(
                {1: (10,)}, {2: ([10], [0.9])}, folds=2
            )


if __name__ == "__main__":
    unittest.main()
