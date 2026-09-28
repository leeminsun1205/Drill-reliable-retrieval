import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drill_ir.data.paths import resolve_data_file


class DataPathTest(unittest.TestCase):
    def test_explicit_existing_path_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "custom.parquet"
            target.touch()
            self.assertEqual(
                resolve_data_file(target, filename="train.parquet"), target.resolve()
            )

    def test_unique_kaggle_discovery_is_used_as_last_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "mounted-dataset" / "train.parquet"
            target.parent.mkdir()
            target.touch()
            with (
                patch("drill_ir.data.paths.LOCAL_DATA_DIR", root / "missing-local"),
                patch("drill_ir.data.paths.KAGGLE_DATASET_DIR", root / "missing-known"),
                patch("drill_ir.data.paths.KAGGLE_INPUT_DIR", root),
            ):
                self.assertEqual(
                    resolve_data_file(None, filename="train.parquet"), target.resolve()
                )

    def test_ambiguous_kaggle_discovery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                target = root / name / "train.parquet"
                target.parent.mkdir()
                target.touch()
            with (
                patch("drill_ir.data.paths.LOCAL_DATA_DIR", root / "missing-local"),
                patch("drill_ir.data.paths.KAGGLE_DATASET_DIR", root / "missing-known"),
                patch("drill_ir.data.paths.KAGGLE_INPUT_DIR", root),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Multiple Kaggle files"):
                    resolve_data_file(None, filename="train.parquet")


if __name__ == "__main__":
    unittest.main()
