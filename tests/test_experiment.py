import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from drill_ir.data import Article, Query, RetrievalResult
from drill_ir.preprocessing import Chunk
from drill_ir.retrieval import ChunkRetrievalResult
from drill_ir.experiment import (
    apply_config_overrides,
    deterministic_split,
    load_config,
    package_score_gap_submission,
    package_submission,
    package_threshold_submission,
    run_submission_inference,
)
from drill_ir.experiment.inference import (
    _read_submission_handoff,
    _write_submission_handoff,
)


class ExperimentTest(unittest.TestCase):
    def test_submission_handoff_preserves_named_split_artifacts(self) -> None:
        public_chunk = Chunk("10:0", 10, "law", 0, "fake", "public", 0, 6, 1)
        private_chunk = Chunk("20:0", 20, "law", 0, "fake", "private", 0, 7, 1)
        first_stage = {
            "public": [RetrievalResult(1, (10,), (0.9,))],
            "private": [RetrievalResult(2, (20,), (0.8,))],
        }
        chunks = {
            "public": [ChunkRetrievalResult(1, (public_chunk,), (0.7,))],
            "private": [ChunkRetrievalResult(2, (private_chunk,), (0.6,))],
        }
        components = {
            "public": {"bm25": [RetrievalResult(1, (10,), (1.0,))]},
            "private": {"bm25": [RetrievalResult(2, (20,), (1.0,))]},
        }
        feedback = {"public": {"1": []}, "private": {"2": []}}
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_submission_handoff(
                Path(directory),
                first_stage=first_stage,
                chunks=chunks,
                components=components,
                feedback=feedback,
                metadata={"cache_hit": True},
            )
            loaded = _read_submission_handoff(manifest)
        loaded_first, loaded_chunks, loaded_components, loaded_feedback, metadata = loaded
        self.assertEqual(loaded_first, first_stage)
        self.assertEqual(loaded_chunks, chunks)
        self.assertEqual(loaded_components, components)
        self.assertEqual(loaded_feedback, feedback)
        self.assertEqual(metadata, {"cache_hit": True})

    def test_direct_chunk_submission_keeps_sparse_state_aligned_per_split(self) -> None:
        class StatefulRetriever:
            def __init__(self):
                self.active_qids = ()
                self.calls = []

            def fit(self, articles):
                pass

            def retrieve(self, queries, top_k):
                self.active_qids = tuple(query.qid for query in queries)
                self.calls.append(("retrieve", self.active_qids))
                return [RetrievalResult(query.qid, (10,), (1.0,)) for query in queries]

            def retrieve_chunks(self, queries, *, top_k, dense_depth):
                qids = tuple(query.qid for query in queries)
                self.calls.append(("chunks", qids))
                if qids != self.active_qids:
                    raise KeyError(qids[0])
                chunk = Chunk("10:0", 10, "law", 0, "fake", "text", 0, 4, 1)
                return [ChunkRetrievalResult(query.qid, (chunk,), (1.0,)) for query in queries]

            def run_metadata(self):
                return {}

        class FakeReranker:
            def __init__(self):
                self.scores = {}

            def rerank_chunks(self, queries, candidates):
                self.scores.update({query.qid: (2.0,) for query in queries})
                return [RetrievalResult(query.qid, (10,), (2.0,)) for query in queries]

            def chunk_scores_by_qid(self):
                return dict(self.scores)

            def run_metadata(self):
                return {}

        retriever = StatefulRetriever()
        query_sets = {
            "public.json": [Query(8631, "public", ())],
            "private.json": [Query(9001, "private", ())],
        }
        config = {
            "method": "hybrid_reranker",
            "seed": 42,
            "data": {"corpus_path": "corpus"},
            "retrieval": {"top_k": 100, "output_k": 1},
            "reranking": {
                "candidate_source": "retrieved_chunks",
                "candidate_pool_size": 50,
                "retrieved_chunk_depth": 100,
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "drill_ir.experiment.inference.load_queries",
            side_effect=lambda path: query_sets[Path(path).name],
        ), patch(
            "drill_ir.experiment.inference.iter_articles",
            return_value=iter([Article(10, "law", "text")]),
        ), patch(
            "drill_ir.experiment.inference.build_retriever", return_value=retriever
        ), patch(
            "drill_ir.experiment.inference.build_reranker", return_value=FakeReranker()
        ):
            run_submission_inference(
                config,
                {"public": "public.json", "private": "private.json"},
                Path(directory) / "output",
            )
        self.assertEqual(
            retriever.calls,
            [
                ("retrieve", (8631,)),
                ("chunks", (8631,)),
                ("retrieve", (9001,)),
                ("chunks", (9001,)),
            ],
        )

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

    def test_selected_main_config_is_direct_chunk_448_64_512(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "configs" / "hybrid_reranker_khoan_diem.yaml"
        )
        self.assertEqual(config["retrieval"]["dense"]["chunking"]["max_tokens"], 448)
        self.assertEqual(config["retrieval"]["dense"]["chunking"]["overlap_tokens"], 64)
        self.assertEqual(config["reranking"]["candidate_source"], "retrieved_chunks")
        self.assertEqual(config["reranking"]["retrieved_chunk_depth"], 100)
        self.assertEqual(config["reranking"]["candidate_pool_size"], 50)
        self.assertEqual(config["reranking"]["max_length"], 512)

    def test_multi_dense_config_uses_selected_peers_without_an_anchor(self) -> None:
        config = load_config(
            Path(__file__).parents[1]
            / "configs"
            / "hybrid_reranker_multi_dense.yaml"
        )
        components = config["retrieval"]["hybrid"]["dense_components"]
        self.assertEqual(
            [item["name"] for item in components],
            ["vietlegal_harrier", "multilingual_e5_large"],
        )
        self.assertEqual(
            config["retrieval"]["dense_profiles"]["vietlegal_harrier"]["encoder"]["model_name"],
            "mainguyen9/vietlegal-harrier-0.6b",
        )
        self.assertEqual(
            config["retrieval"]["dense_profiles"]["multilingual_e5_large"]["encoder"]["model_name"],
            "intfloat/multilingual-e5-large",
        )
        self.assertEqual(
            config["retrieval"]["dense"]["chunking"]["tokenizer_model_name"],
            "BAAI/bge-m3",
        )
        self.assertEqual(config["retrieval"]["dense"]["search"]["backend"], "faiss_flat")
        self.assertEqual(config["retrieval"]["dense"]["search"]["device"], "cpu")
        self.assertEqual(config["retrieval"]["bm25"]["backend"], "bm25s")
        self.assertEqual(config["reranking"]["batching"], "per_query")
        self.assertEqual(config["reranking"]["outer_pair_batch_size"], 1024)
        self.assertNotIn("device", config["reranking"])
        self.assertEqual(config["reranking"]["batch_size"], 8)

    def test_sparse_screening_config_fixes_matched_bm25_protocol(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "configs" / "sparse_screening.yaml"
        )
        bm25 = config["retrieval"]["bm25"]
        self.assertEqual(config["method"], "bm25")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["retrieval"]["top_k"], 100)
        self.assertEqual(bm25["variant"], "bm25")
        self.assertEqual(bm25["delta"], 1.0)
        self.assertEqual(bm25["preprocessing"]["tokenizer"]["word_segmenter"], "vncorenlp")
        self.assertEqual(config["evaluation"]["output_k_values"], [1, 2, 3, 4, 5, 10])
        self.assertEqual(config["evaluation"]["recall_at_k"], [1, 5, 10, 20, 50, 100])

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

    def test_score_gap_submission_uses_adaptive_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            rows = [
                {"qid": 1, "retrieved_aids": [10, 20, 30], "scores": [0.9, 0.8, 0.1]},
                {"qid": 2, "retrieved_aids": [40, 50, 60], "scores": [0.9, 0.85, 0.8]},
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            destination = package_score_gap_submission(
                predictions, {1, 2}, root / "gap_results.zip",
                minimum_gap=0.2, fallback_k=1, maximum_k=3,
            )
            with ZipFile(destination) as archive:
                payload = json.loads(archive.read("results.json"))
            self.assertEqual(
                payload,
                [
                    {"qid": 1, "relevant_laws": [10, 20]},
                    {"qid": 2, "relevant_laws": [40]},
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
            manifest = json.loads((output / "submission_manifest.json").read_text())
            self.assertEqual(len(manifest), 6)
            self.assertEqual(
                {item["file"] for item in manifest},
                {
                    "public_results.zip",
                    "private_results.zip",
                    "public_threshold_0_fallback2_results.zip",
                    "private_threshold_0_fallback2_results.zip",
                    "public_threshold_0p5_fallback2_results.zip",
                    "private_threshold_0p5_fallback2_results.zip",
                },
            )

    def test_submission_threshold_arguments_fail_before_expensive_work(self) -> None:
        minimal = {"method": "bm25"}
        with self.assertRaisesRegex(ValueError, "either score_threshold"):
            run_submission_inference(
                minimal,
                {"public": "unused"},
                "unused",
                score_threshold=1.0,
                score_thresholds=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            run_submission_inference(
                minimal,
                {"public": "unused"},
                "unused",
                score_thresholds=[float("nan")],
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_submission_inference(
                minimal,
                {"public": "unused"},
                "unused",
                score_thresholds=[1.0],
                threshold_fallback_k=0,
            )


if __name__ == "__main__":
    unittest.main()
