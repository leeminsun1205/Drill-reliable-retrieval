import unittest

from drill_ir.analysis.cardinality_screen import apply_fitted_case, evaluate_nested_case


class CardinalityScreenTest(unittest.TestCase):
    def _fixture(self):
        gold, rankings = {}, {}
        for qid in range(1, 21):
            first = qid * 10
            if qid % 2:
                gold[qid] = (first,); scores = [0.95, 0.20, 0.10, 0.05, 0.01]
            else:
                gold[qid] = (first, first + 1); scores = [0.95, 0.90, 0.10, 0.05, 0.01]
            rankings[qid] = ([first + offset for offset in range(5)], scores)
        return gold, rankings

    def test_ridge_nested_and_deployment(self):
        gold, rankings = self._fixture()
        metrics, params, _, fitted = evaluate_nested_case("utility_ridge", {"alpha": [1.0, 10.0]}, rankings, gold, outer_folds=5, inner_folds=5, seed=42, maximum_k=5)
        self.assertEqual(metrics["f2"], 1.0)
        self.assertIn(params["alpha"], {1.0, 10.0})
        selected, _ = apply_fitted_case("utility_ridge", fitted, rankings, 5)
        self.assertEqual(selected, {qid: list(values) for qid, values in gold.items()})

    def test_multiclass_logistic_nested(self):
        gold, rankings = self._fixture()
        metrics, _, _, _ = evaluate_nested_case("multiclass_logistic", {"C": [1.0], "class_weight": [None, "balanced"]}, rankings, gold, outer_folds=5, inner_folds=5, seed=42, maximum_k=5)
        self.assertEqual(metrics["f2"], 1.0)


if __name__ == "__main__": unittest.main()
