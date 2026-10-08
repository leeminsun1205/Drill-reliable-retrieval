import json
from pathlib import Path
import tempfile
import unittest

from drill_ir.analysis import build_ltr_examples
from drill_ir.data import Query


class LtrFeatureTest(unittest.TestCase):
    def test_builds_aligned_features_from_saved_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, rows: list[dict]) -> None:
                (root / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            write(
                "predictions.jsonl",
                [{"qid": 1, "retrieved_aids": [10, 20], "scores": [4.0, 2.0]}],
            )
            write(
                "first_stage_predictions.jsonl",
                [{"qid": 1, "retrieved_aids": [20, 10], "scores": [0.4, 0.3]}],
            )
            write(
                "component_rankings.jsonl",
                [
                    {
                        "qid": 1,
                        "components": {
                            "sparse": {
                                "retrieved_aids": [10],
                                "scores": [3.0],
                            },
                            "dense": {
                                "retrieved_aids": [20, 10],
                                "scores": [0.9, 0.8],
                            },
                        },
                    }
                ],
            )
            write(
                "reranker_chunk_candidates.jsonl",
                [
                    {
                        "qid": 1,
                        "chunks": [
                            {"aid": 10, "fusion_score": 0.3},
                            {"aid": 10, "fusion_score": 0.2},
                            {"aid": 20, "fusion_score": 0.4},
                        ],
                    }
                ],
            )
            examples = build_ltr_examples(
                root, [Query(1, "alpha beta", (10,))]
            )
            self.assertEqual(examples["features"].shape, (2, 13))
            self.assertEqual(examples["labels"].tolist(), [1, 0])
            self.assertEqual(examples["qids"].tolist(), [1, 1])
            self.assertEqual(examples["aids"].tolist(), [10, 20])
            chunk_count_index = examples["feature_names"].index("chunk_count")
            self.assertEqual(examples["features"][0, chunk_count_index], 2.0)
            sparse_rr_index = examples["feature_names"].index(
                "sparse_reciprocal_rank"
            )
            self.assertEqual(examples["features"][1, sparse_rr_index], 0.0)

    def test_builds_prefixed_submission_features_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, row: dict) -> None:
                (root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

            write("public_predictions.jsonl", {"qid": 7, "retrieved_aids": [10], "scores": [3.0]})
            write("public_first_stage_predictions.jsonl", {"qid": 7, "retrieved_aids": [10], "scores": [0.4]})
            write(
                "public_component_rankings.jsonl",
                {"qid": 7, "components": {"sparse": {"retrieved_aids": [10], "scores": [2.0]}, "dense": {"retrieved_aids": [10], "scores": [0.8]}}},
            )
            write(
                "public_reranker_chunk_candidates.jsonl",
                {"qid": 7, "chunks": [{"aid": 10, "fusion_score": 0.3}]},
            )
            examples = build_ltr_examples(
                root, [Query(7, "test query", ())], prefix="public_"
            )
            self.assertEqual(examples["features"].shape, (1, 13))
            self.assertEqual(examples["labels"].tolist(), [0])

    def test_builds_separate_features_for_multiple_dense_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, row: dict) -> None:
                (root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

            write(
                "predictions.jsonl",
                {"qid": 1, "retrieved_aids": [10], "scores": [4.0]},
            )
            write(
                "first_stage_predictions.jsonl",
                {"qid": 1, "retrieved_aids": [10], "scores": [0.4]},
            )
            write(
                "component_rankings.jsonl",
                {
                    "qid": 1,
                    "components": {
                        "sparse": {"retrieved_aids": [10], "scores": [2.0]},
                        "dense:bge_m3": {
                            "retrieved_aids": [10],
                            "scores": [0.8],
                        },
                        "dense:vietlegal_e5": {
                            "retrieved_aids": [10],
                            "scores": [0.9],
                        },
                    },
                },
            )
            write(
                "reranker_chunk_candidates.jsonl",
                {"qid": 1, "chunks": [{"aid": 10, "fusion_score": 0.3}]},
            )
            examples = build_ltr_examples(root, [Query(1, "query", (10,))])
            self.assertEqual(examples["features"].shape, (1, 15))
            self.assertIn("dense_bge_m3_score", examples["feature_names"])
            self.assertIn("dense_vietlegal_e5_reciprocal_rank", examples["feature_names"])
            self.assertEqual(
                examples["features"][0, examples["feature_names"].index("dense_vietlegal_e5_score")],
                0.9,
            )


if __name__ == "__main__":
    unittest.main()
