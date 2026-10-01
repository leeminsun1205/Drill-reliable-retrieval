import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from drill_ir.data import Query
from drill_ir.experiment import (
    apply_config_overrides,
    deterministic_split,
    load_config,
    package_submission,
    package_threshold_submission,
    run_submission_inference,
)


class ExperimentTest(unittest.TestCase):
    def test_hybrid_config_is_supported(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "hybrid.yaml")
        self.assertEqual(config["method"], "hybrid")
        self.assertEqual(config["retrieval"]["hybrid"]["fusion"], "weighted_rrf")

    def test_hybrid_reranker_config_is_supported(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "configs" / "hybrid_reranker.yaml"
        )
        self.assertEqual(config["method"], "hybrid_reranker")
        self.assertEqual(config["reranking"]["model_name"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(config["evaluation"]["output_k_values"], [1, 2, 3, 4, 5])

    def test_config_overrides_are_typed_and_do_not_mutate_source(self) -> None:
        source = {"retrieval": {"top_k": 100, "prefix": "query: "}}
        resolved = apply_config_overrides(
            source,
            ["retrieval.top_k=50", "retrieval.prefix="],
        )
        self.assertEqual(resolved["retrieval"], {"top_k": 50, "prefix": ""})
        self.assertEqual(source["retrieval"]["top_k"], 100)

    def test_config_overrides_reject_unknown_paths(self) -> None:
        with self.assertRaises(ValueError):
            apply_config_overrides({"retrieval": {"top_k": 100}}, ["retrieval.typo=5"])

    def test_split_is_deterministic_and_preserves_input_order(self) -> None:
        queries = [Query(qid=i, question=str(i), relevant_aids=(i,)) for i in range(10)]
        train_a, dev_a = deterministic_split(queries, dev_fraction=0.2, seed=42)
        train_b, dev_b = deterministic_split(queries, dev_fraction=0.2, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(dev_a, dev_b)
        self.assertEqual([q.qid for q in train_a], sorted(q.qid for q in train_a))
        self.assertEqual(len(dev_a), 2)

    def test_submission_zip_contains_only_results_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"qid": 1, "retrieved_aids": [10, 10, 20], "scores": [3, 2, 1]})
                + "\n",
                encoding="utf-8",
            )
            destination = package_submission(
                predictions, {1}, root / "results.zip", output_k=3
            )
            with ZipFile(destination) as archive:
                self.assertEqual(archive.namelist(), ["results.json"])
                payload = json.loads(archive.read("results.json"))
            self.assertEqual(payload, [{"qid": 1, "relevant_laws": [10, 20]}])

    def test_threshold_submission_uses_empty_only_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            rows = [
                {"qid": 1, "retrieved_aids": [10, 20, 30], "scores": [5.0, 4.1, 3.0]},
                {"qid": 2, "retrieved_aids": [40, 50, 60], "scores": [3.9, 3.0, 2.0]},
                {"qid": 3, "retrieved_aids": [70, 80, 90], "scores": [4.2, 3.0, 2.0]},
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            destination = package_threshold_submission(
                predictions,
                {1, 2, 3},
                root / "threshold_results.zip",
                score_threshold=4.0,
                fallback_k=2,
            )
            with ZipFile(destination) as archive:
                payload = json.loads(archive.read("results.json"))
            self.assertEqual(
                payload,
                [
                    {"qid": 1, "relevant_laws": [10, 20]},
                    {"qid": 2, "relevant_laws": [40, 50]},
                    {"qid": 3, "relevant_laws": [70]},
                ],
            )

    def test_bm25_inference_creates_public_and_private_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.parquet"
            schema = pa.schema(
                [
                    ("law_id", pa.string()),
                    (
                        "content",
                        pa.list_(
                            pa.struct(
                                [("aid", pa.int64()), ("content_Article", pa.string())]
                            )
                        ),
                    ),
                ]
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "law_id": "law-a",
                            "content": [
                                {"aid": 10, "content_Article": "học văn hóa xóa mù chữ"},
                                {"aid": 20, "content_Article": "tài khoản ngoại tệ"},
                            ],
                        }
                    ],
                    schema=schema,
                ),
                corpus,
            )
            public = root / "public.json"
            private = root / "private.json"
            public.write_text(
                json.dumps([{"id": 1, "question": "tài khoản ngoại tệ"}]),
                encoding="utf-8",
            )
            private.write_text(
                json.dumps([{"qid": 2, "question": "xóa mù chữ"}]),
                encoding="utf-8",
            )
            config = {
                "experiment_name": "submission-test",
                "method": "bm25",
                "seed": 42,
                "data": {"corpus_path": str(corpus)},
                "retrieval": {
                    "top_k": 2,
                    "output_k": 1,
                    "bm25": {
                        "preprocessing": {"tokenizer": {"word_segmenter": "none"}}
                    },
                },
                "runtime": {"accelerator": "cpu", "devices": "auto"},
                "output": {"root_dir": str(root)},
            }
            output = run_submission_inference(
                config,
                {"public": public, "private": private},
                root / "submissions",
                score_thresholds=[0.0, 0.5],
                threshold_fallback_k=2,
            )
            for name, expected_qid in (("public", 1), ("private", 2)):
                with ZipFile(output / f"{name}_results.zip") as archive:
                    payload = json.loads(archive.read("results.json"))
                self.assertEqual(payload[0]["qid"], expected_qid)
                self.assertEqual(len(payload[0]["relevant_laws"]), 1)
                self.assertTrue(
                    (output / f"{name}_threshold_0_fallback2_results.zip").is_file()
                )
                self.assertTrue(
                    (output / f"{name}_threshold_0p5_fallback2_results.zip").is_file()
                )
            run_info = json.loads((output / "run_info.json").read_text())
            self.assertEqual(run_info["threshold_sweep"]["thresholds"], [0.0, 0.5])


if __name__ == "__main__":
    unittest.main()
