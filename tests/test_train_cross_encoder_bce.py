import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_cross_encoder_bce.py"
SPEC = importlib.util.spec_from_file_location("train_cross_encoder_bce", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_pairs_accepts_strict_binary_schema(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {"query": "mức phạt", "passage": "Điều 1", "label": 1},
            {"query": "mức phạt", "passage": "Điều 2", "label": 0},
        ],
    )

    pairs = MODULE._load_pairs(path)

    assert [pair["label"] for pair in pairs] == [1.0, 0.0]


def test_link_final_model_reuses_checkpoint_storage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    source = checkpoint / "model.safetensors"
    source.write_bytes(b"weights")

    MODULE._link_final_model(checkpoint, checkpoint.parent)

    final_weight = checkpoint.parent / "model.safetensors"
    assert final_weight.read_bytes() == b"weights"
    assert final_weight.stat().st_ino == source.stat().st_ino


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                {"query": "q", "passage": "p", "label": 1, "qid": 7},
                {"query": "q", "passage": "n", "label": 0},
            ],
            "expected exactly",
        ),
        (
            [
                {"query": "q", "passage": "p", "label": 2},
                {"query": "q", "passage": "n", "label": 0},
            ],
            "label must be binary",
        ),
        (
            [
                {"query": "q", "passage": "p", "label": 1},
                {"query": "q2", "passage": "p2", "label": 1},
            ],
            "both positive and negative",
        ),
    ],
)
def test_load_pairs_rejects_invalid_training_artifacts(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match=message):
        MODULE._load_pairs(path)
