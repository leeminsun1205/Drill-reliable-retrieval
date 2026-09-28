import unittest

from drill_ir.evaluation import evaluate_predictions, macro_recall_at_k


class DrillMetricsTest(unittest.TestCase):
    def test_macro_precision_recall_then_f2(self) -> None:
        metrics = evaluate_predictions(
            gold={1: [2, 5], 2: [7]},
            predictions={1: [2, 3, 5], 2: []},
        )
        self.assertAlmostEqual(metrics.precision, 1 / 3)
        self.assertAlmostEqual(metrics.recall, 1 / 2)
        self.assertAlmostEqual(metrics.f2, 5 / 11)

    def test_duplicate_ids_do_not_change_set_metrics(self) -> None:
        metrics = evaluate_predictions(gold={1: [2, 2, 5]}, predictions={1: [2, 2, 5]})
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f2, 1.0)

    def test_unknown_prediction_qid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions(gold={1: [2]}, predictions={1: [2], 9: [3]})

    def test_macro_recall_at_k(self) -> None:
        recalls = macro_recall_at_k(
            gold={1: [2, 5], 2: [7]},
            rankings={1: [2, 3, 5], 2: [8, 7, 9]},
            cutoffs=[1, 2, 3],
        )
        self.assertAlmostEqual(recalls[1], 1 / 4)
        self.assertAlmostEqual(recalls[2], 3 / 4)
        self.assertEqual(recalls[3], 1.0)

    def test_macro_recall_at_k_missing_ranking_is_zero(self) -> None:
        recalls = macro_recall_at_k(
            gold={1: [2], 2: [7]}, rankings={1: [2]}, cutoffs=[1]
        )
        self.assertEqual(recalls[1], 0.5)

    def test_macro_recall_at_k_rejects_invalid_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            macro_recall_at_k(gold={1: [2]}, rankings={1: [2]}, cutoffs=[0])
        with self.assertRaises(ValueError):
            macro_recall_at_k(gold={1: [2]}, rankings={1: [2]}, cutoffs=[1.5])


if __name__ == "__main__":
    unittest.main()
