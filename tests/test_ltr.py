import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from drill_ir.analysis import (
    build_ltr_examples,
    drop_ltr_features,
    limit_ltr_examples,
    ltr_feature_groups,
)
from drill_ir.data import Query


class LtrFeatureTest(unittest.TestCase):
    def test_feature_groups_are_complete_and_support_column_ablation(self) -> None:
        examples = {
            "feature_names": (
                "reranker_score",
                "chunk_reranker_mean",
                "article_character_count",
                "query_chunk_overlap_mean",
            ),
            "features": np.asarray([[1.0, 2.0, 3.0, 4.0]]),
            "labels": np.asarray([1]),
            "qids": np.asarray([1]),
            "aids": np.asarray([10]),
            "original_ranks": np.asarray([1]),
            "feature_schema": "enhanced_v3",
        }
        groups = ltr_feature_groups(examples["feature_names"])
        grouped_names = [name for names in groups.values() for name in names]
        self.assertCountEqual(grouped_names, examples["feature_names"])
        reduced = drop_ltr_features(
            examples, groups["chunk_reranker_distribution"]
        )
        self.assertEqual(reduced["features"].shape, (1, 3))
        self.assertNotIn("chunk_reranker_mean", reduced["feature_names"])
        self.assertEqual(reduced["feature_schema"], "enhanced_v3")

    def test_limits_rows_by_saved_reranker_rank(self) -> None:
        examples = {
            "feature_names": ("score",),
            "features": np.asarray([[3.0], [2.0], [1.0]]),
            "labels": np.asarray([1, 0, 1]),
            "qids": np.asarray([1, 1, 2]),
            "aids": np.asarray([10, 20, 30]),
            "original_ranks": np.asarray([1, 2, 1]),
        }
        limited = limit_ltr_examples(examples, 1)
        self.assertEqual(limited["aids"].tolist(), [10, 30])
        self.assertEqual(limited["labels"].tolist(), [1, 1])
        self.assertEqual(limited["feature_names"], ("score",))
        self.assertIs(limit_ltr_examples(examples, None), examples)
        with self.assertRaises(ValueError):
            limit_ltr_examples(examples, 0)

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
            self.assertEqual(examples["features"].shape, (2, 39))
            self.assertEqual(examples["labels"].tolist(), [1, 0])
            self.assertEqual(examples["qids"].tolist(), [1, 1])
            self.assertEqual(examples["aids"].tolist(), [10, 20])
            chunk_count_index = examples["feature_names"].index("chunk_count")
            self.assertEqual(examples["features"][0, chunk_count_index], 2.0)
            sparse_rr_index = examples["feature_names"].index(
                "sparse_reciprocal_rank"
            )
            self.assertEqual(examples["features"][1, sparse_rr_index], 0.0)
            self.assertEqual(
                examples["features"][0, examples["feature_names"].index("retriever_hit_count")],
                2.0,
            )
            self.assertAlmostEqual(
                examples["features"][0, examples["feature_names"].index("chunk_fusion_mean")],
                0.25,
            )

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
            self.assertEqual(examples["features"].shape, (1, 39))
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
            self.assertEqual(examples["features"].shape, (1, 41))
            self.assertIn("dense_bge_m3_score", examples["feature_names"])
            self.assertIn("dense_vietlegal_e5_reciprocal_rank", examples["feature_names"])
            self.assertEqual(
                examples["features"][0, examples["feature_names"].index("dense_vietlegal_e5_score")],
                0.9,
            )
            base_examples = build_ltr_examples(
                root,
                [Query(1, "query", (10,))],
                feature_schema="base_v1",
            )
            self.assertEqual(base_examples["features"].shape, (1, 15))
            self.assertEqual(len(base_examples["feature_names"]), 15)
            self.assertNotIn("reranker_score_z", base_examples["feature_names"])
            self.assertEqual(base_examples["feature_schema"], "base_v1")

    def test_adds_reference_provenance_without_using_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, payload: object, *, jsonl: bool = True) -> None:
                text = json.dumps(payload)
                (root / name).write_text(text + ("\n" if jsonl else ""), encoding="utf-8")

            write("predictions.jsonl", {"qid": 1, "retrieved_aids": [20], "scores": [2.0]})
            write("first_stage_predictions.jsonl", {"qid": 1, "retrieved_aids": [10, 20], "scores": [0.5, 0.4]})
            write("component_rankings.jsonl", {"qid": 1, "components": {"sparse": {"retrieved_aids": [10], "scores": [1.0]}, "dense": {"retrieved_aids": [20], "scores": [0.8]}}})
            write("reranker_chunk_candidates.jsonl", {"qid": 1, "chunks": [{"aid": 20, "fusion_score": 0.4, "chunk_type": "clause"}]})
            write(
                "retrieval_feedback.json",
                {"explicit_references": [{"qid": 1, "seed_aid": 10, "reference_aid": 20, "scope": "same_document", "base_rank": 2}]},
                jsonl=False,
            )
            examples = build_ltr_examples(root, [Query(1, "Theo Điều 3", (20,))])
            row = examples["features"][0]
            names = examples["feature_names"]
            self.assertEqual(row[names.index("is_reference_expanded")], 1.0)
            self.assertEqual(row[names.index("reference_base_reciprocal_rank")], 0.5)
            self.assertEqual(row[names.index("reference_seed_first_stage_reciprocal_rank")], 1.0)
            self.assertEqual(row[names.index("query_has_explicit_legal_reference")], 1.0)

    def test_enhanced_v2_adds_persisted_chunk_neural_and_lexical_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, row: dict) -> None:
                (root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

            write("predictions.jsonl", {"qid": 1, "retrieved_aids": [10], "scores": [4.0]})
            write("first_stage_predictions.jsonl", {"qid": 1, "retrieved_aids": [10], "scores": [0.4]})
            write(
                "component_rankings.jsonl",
                {"qid": 1, "components": {
                    "sparse": {"retrieved_aids": [10], "scores": [2.0]},
                    "dense:harrier": {"retrieved_aids": [10], "scores": [0.8]},
                    "dense:e5": {"retrieved_aids": [10], "scores": [0.9]},
                }},
            )
            base = {
                "aid": 10,
                "chunk_type": "diem",
                "fusion_score": 0.3,
                "character_count": 100,
                "normalized_token_count": 20,
                "query_chunk_overlap_count": 3,
                "query_token_coverage": 0.75,
                "chunk_token_coverage": 0.15,
                "legal_citation_overlap_count": 1,
            }
            write(
                "reranker_chunk_candidates.jsonl",
                {"qid": 1, "chunks": [
                    {**base, "chunk_id": "10:0", "chunk_index": 0, "reranker_score": 4.0},
                    {**base, "chunk_id": "10:1", "chunk_index": 1, "reranker_score": 2.0,
                     "character_count": 80, "normalized_token_count": 16},
                ]},
            )

            examples = build_ltr_examples(
                root,
                [Query(1, "Điều 3 quy định gì", (10,))],
                feature_schema="enhanced_v2",
            )

            self.assertEqual(examples["features"].shape, (1, 61))
            names = examples["feature_names"]
            row = examples["features"][0]
            self.assertEqual(row[names.index("chunk_reranker_mean")], 3.0)
            self.assertEqual(row[names.index("chunk_reranker_top1_top2_gap")], 2.0)
            self.assertEqual(row[names.index("query_token_coverage_max")], 0.75)
            self.assertEqual(row[names.index("legal_citation_overlap_max")], 1.0)

            static = {
                "aid": 10,
                "article_character_count": 1000,
                "article_whitespace_token_count": 200,
                "article_normalized_token_count": 210,
                "article_unique_token_count": 120,
                "article_clause_marker_count": 3,
                "article_point_marker_count": 5,
                "article_legal_citation_count": 2,
                "article_digit_token_fraction": 0.1,
            }
            write("static.jsonl", static)
            v3 = build_ltr_examples(
                root,
                [Query(1, "Điều 3 quy định gì", (10,))],
                feature_schema="enhanced_v3",
                static_article_features=root / "static.jsonl",
            )
            self.assertEqual(v3["features"].shape, (1, 69))
            self.assertEqual(
                v3["features"][0, v3["feature_names"].index("article_character_count")],
                1000,
            )


if __name__ == "__main__":
    unittest.main()
