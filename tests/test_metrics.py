import unittest

from drill_ir.evaluation import evaluate_predictions


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


if __name__ == "__main__":
    unittest.main()
