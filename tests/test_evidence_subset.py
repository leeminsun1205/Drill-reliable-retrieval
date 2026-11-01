import unittest

from drill_ir.analysis.evidence_subset import (
    apply_subset_case,
    evaluate_subset_case,
    oracle_subset_metrics,
)


class EvidenceSubsetTest(unittest.TestCase):
    def fixture(self):
        gold = {}
        rankings = {}
        for qid in range(1, 21):
            first = qid * 10
            scores = [0.95, 0.80, 0.65, 0.20, 0.05]
            # Rank two is a distractor; the relevant set is non-prefix {1, 3}.
            gold[qid] = (first, first + 2)
            rankings[qid] = ([first + offset for offset in range(5)], scores)
        return gold, rankings

    def test_oracle_subset_exposes_non_prefix_headroom(self):
        gold, rankings = self.fixture()
        metrics, prefix, subset = oracle_subset_metrics(rankings, gold, 5)
        self.assertGreater(metrics["oracle_subset"]["f2"], metrics["oracle_prefix"]["f2"])
        self.assertEqual(metrics["oracle_subset"]["f2"], 1.0)
        self.assertEqual(subset[1], [10, 12])
        self.assertEqual(prefix[1], [10, 11, 12])

    def test_both_cases_are_fold_safe_and_serializable(self):
        gold, rankings = self.fixture()
        for case, grid in (
            ("cardinality_binary_subset", [0.1, 1.0]),
            ("joint_subset_utility", [0.1, 1.0]),
        ):
            metrics, deployment, fitted, predictions, diagnostics = evaluate_subset_case(
                case,
                rankings,
                gold,
                outer_folds=5,
                inner_folds=5,
                seed=42,
                maximum_k=5,
                C_multi=0.1,
                grid=grid,
            )
            self.assertEqual(set(predictions), set(gold))
            self.assertEqual(set(diagnostics), set(gold))
            self.assertIn(deployment["selected_parameter"], grid)
            self.assertGreaterEqual(metrics["f2"], 0.0)
            self.assertLessEqual(metrics["f2"], 1.0)
            selected, details = apply_subset_case(case, fitted, rankings, 5)
            self.assertEqual(set(selected), set(gold))
            self.assertTrue(all(1 <= len(value) <= 5 for value in selected.values()))
            self.assertTrue(all("non_prefix" in value for value in details.values()))


if __name__ == "__main__":
    unittest.main()
