import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from drill_ir.data import Query
from drill_ir.experiment import (
    apply_config_overrides,
    deterministic_split,
    package_submission,
)


class ExperimentTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
