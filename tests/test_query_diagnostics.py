import json
from pathlib import Path
import tempfile
import unittest

from drill_ir.analysis import analyze_query_retrieval, write_query_diagnostics
from drill_ir.data import Query


class QueryDiagnosticsTest(unittest.TestCase):
    def test_attributes_macro_recall_gap_by_query_component_and_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, rows: list[dict]) -> None:
                (root / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )

            write("first_stage_predictions.jsonl", [
                {"qid": 1, "retrieved_aids": [10, 20], "scores": [2.0, 1.0]},
                {"qid": 2, "retrieved_aids": [40], "scores": [1.0]},
            ])
            write("component_rankings.jsonl", [
                {"qid": 1, "components": {
                    "sparse": {"retrieved_aids": [10], "scores": [1.0]},
                    "dense:model_a": {"retrieved_aids": [20], "scores": [1.0]},
                }},
                {"qid": 2, "components": {
                    "sparse": {"retrieved_aids": [40], "scores": [1.0]},
                    "dense:model_a": {"retrieved_aids": [50], "scores": [1.0]},
                }},
            ])
            write("reranker_chunk_candidates.jsonl", [
                {"qid": 1, "chunks": [{"aid": 10}, {"aid": 10}]},
                {"qid": 2, "chunks": [{"aid": 40}]},
            ])
            write("predictions.jsonl", [
                {"qid": 1, "retrieved_aids": [10, 30], "scores": [2.0, 1.0]},
                {"qid": 2, "retrieved_aids": [40], "scores": [1.0]},
            ])
            queries = [
                Query(1, "Điều 1", (10, 20)),
                Query(2, "ordinary question", (30,)),
            ]
            report = analyze_query_retrieval(root, queries)
            fused50 = next(
                row for row in report["stage_metrics"]
                if row["stage"] == "fused_first_stage" and row["cutoff"] == 50
            )
            self.assertEqual(fused50["macro_recall"], 0.5)
            self.assertEqual(fused50["complete_coverage_queries"], 1)
            self.assertEqual(fused50["zero_coverage_queries"], 1)
            self.assertEqual(report["component_union_oracle_macro_recall_at_50"], 0.5)
            self.assertEqual(
                report["component_attribution_at_50"]["sparse"]["unique_gold_at_50"], 1
            )
            self.assertEqual(
                report["component_attribution_at_50"]["dense:model_a"]["unique_gold_at_50"], 1
            )
            direct50 = next(
                row for row in report["stage_metrics"]
                if row["stage"] == "direct_chunk_candidates" and row["cutoff"] == 50
            )
            self.assertEqual(direct50["macro_recall"], 0.25)
            self.assertEqual(
                report["queries"][0]["gold_attribution"]["20"][
                    "found_by_component_but_missing_fused_at_50"
                ],
                False,
            )
            self.assertEqual(report["hard_queries"][0]["qid"], 2)
            self.assertAlmostEqual(
                sum(row["macro_recall50_gap_contribution"] for row in report["queries"]),
                1.0 - fused50["macro_recall"],
            )
            output = root / "analysis"
            write_query_diagnostics(report, output)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "query_diagnostics.jsonl").is_file())
            self.assertTrue((output / "stage_metrics.csv").is_file())
            self.assertTrue((output / "hard_queries.csv").is_file())


if __name__ == "__main__":
    unittest.main()
