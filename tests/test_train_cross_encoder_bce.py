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


def test_load_pairwise_rows_accepts_strict_ranking_schema(tmp_path: Path) -> None:
    path = tmp_path / "pairwise.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query": "mức phạt",
                "positive": "Điều đúng",
                "negative": "Điều gây nhiễu",
            }
        ],
    )

    rows = MODULE._load_pairwise_rows(path)

    assert rows == [
        {
            "negative": "Điều gây nhiễu",
            "positive": "Điều đúng",
            "query": "mức phạt",
        }
    ]


def test_load_pairwise_rows_rejects_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "pairwise.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query": "q",
                "positive": "p",
                "negative": "n",
                "qid": 7,
            }
        ],
    )

    with pytest.raises(ValueError, match="expected exactly"):
        MODULE._load_pairwise_rows(path)


def test_seeded_passage_index_is_reproducible_and_bounded() -> None:
    first = MODULE._seeded_passage_index(
        seed=42, epoch=2, row_index=17, field="passages", passage_count=8
    )
    second = MODULE._seeded_passage_index(
        seed=42, epoch=2, row_index=17, field="passages", passage_count=8
    )
    assert first == second
    assert 0 <= first < 8


def test_seeded_passage_index_uses_epoch_and_field_identity() -> None:
    epoch_indices = {
        MODULE._seeded_passage_index(
            seed=42, epoch=epoch, row_index=17, field="passages", passage_count=97
        )
        for epoch in range(5)
    }
    assert len(epoch_indices) > 1
    assert MODULE._seeded_passage_index(
        seed=42,
        epoch=0,
        row_index=17,
        field="positive_passages",
        passage_count=97,
    ) != MODULE._seeded_passage_index(
        seed=42,
        epoch=0,
        row_index=17,
        field="negative_passages",
        passage_count=97,
    )


def test_seeded_passage_index_rejects_empty_bag() -> None:
    with pytest.raises(ValueError, match="passage_count must be positive"):
        MODULE._seeded_passage_index(
            seed=42, epoch=0, row_index=0, field="passages", passage_count=0
        )


def test_seeded_cycle_covers_bag_before_repeating() -> None:
    selections = [
        MODULE._seeded_cycle_passage_indices(
            seed=42,
            epoch=epoch,
            row_index=17,
            field="passages",
            passage_count=10,
            max_passages=4,
        )
        for epoch in range(3)
    ]
    assert [len(selection) for selection in selections] == [4, 4, 2]
    assert len(set().union(*map(set, selections))) == 10
    assert sum(len(selection) for selection in selections) == 10


def test_seeded_cycle_is_reproducible_and_restarts_with_new_order() -> None:
    first_cycle = [
        MODULE._seeded_cycle_passage_indices(
            seed=42,
            epoch=epoch,
            row_index=17,
            field="passages",
            passage_count=8,
            max_passages=4,
        )
        for epoch in range(2)
    ]
    repeated = [
        MODULE._seeded_cycle_passage_indices(
            seed=42,
            epoch=epoch,
            row_index=17,
            field="passages",
            passage_count=8,
            max_passages=4,
        )
        for epoch in range(2)
    ]
    second_cycle = MODULE._seeded_cycle_passage_indices(
        seed=42,
        epoch=2,
        row_index=17,
        field="passages",
        passage_count=8,
        max_passages=4,
    )
    assert first_cycle == repeated
    assert second_cycle != first_cycle[0]


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
