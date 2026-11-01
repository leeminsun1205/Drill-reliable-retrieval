import unittest

from drill_ir.analysis import (
    apply_stopping_policy,
    evaluate_nested_stopping_policy,
    fit_deployment_stopping_policy,
)


class StoppingPolicyTest(unittest.TestCase):
    def _fixture(self):
        gold = {}
        rankings = {}
        for qid in range(1, 21):
            first = qid * 10
            if qid % 2:
                gold[qid] = (first,)
                scores = [0.95, 0.20, 0.10, 0.05, 0.01]
            else:
                gold[qid] = (first, first + 1)
                scores = [0.95, 0.90, 0.10, 0.05, 0.01]
            rankings[qid] = ([first + offset for offset in range(5)], scores)
        return gold, rankings

    def test_nested_stopping_is_query_grouped_and_covers_all_queries(self):
        gold, rankings = self._fixture()
        result = evaluate_nested_stopping_policy(
            gold, rankings, folds=5, seed=42, maximum_k=5
        )
        self.assertEqual(result["queries"], 20)
        self.assertEqual(len(result["folds"]), 5)
        self.assertEqual(result["f2"], 1.0)
        self.assertEqual(result["output_size"]["minimum"], 1)
        self.assertEqual(result["output_size"]["maximum"], 2)

    def test_deployment_policy_serializes_and_applies_without_gold(self):
        gold, rankings = self._fixture()
        policy = fit_deployment_stopping_policy(
            gold, rankings, folds=5, seed=42, maximum_k=5
        )
        selected = apply_stopping_policy(policy, rankings)
        self.assertEqual(set(selected), set(gold))
        self.assertTrue(all(1 <= len(aids) <= 5 for aids in selected.values()))
        self.assertIn(policy["model"]["mode"], {"constant", "logistic"})

    def test_rejects_non_descending_scores(self):
        with self.assertRaisesRegex(ValueError, "descending"):
            evaluate_nested_stopping_policy(
                {1: (10,), 2: (20,)},
                {1: ([10, 11], [0.1, 0.2]), 2: ([20, 21], [0.9, 0.1])},
                folds=2,
            )


if __name__ == "__main__":
    unittest.main()
