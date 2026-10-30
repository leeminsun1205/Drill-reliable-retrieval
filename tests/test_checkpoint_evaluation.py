import hashlib
import json
from pathlib import Path

import yaml

from drill_ir.experiment.checkpoint_evaluation import (
    REQUIRED_RESULTS,
    build_cases,
    evaluate_selected_checkpoint,
    expected_contract,
    load_and_validate_handoff,
)


def _config(pool_size: int) -> dict:
    contract = expected_contract(pool_size)
    return {
        "retrieval": {
            "hybrid": {
                "rrf_k": contract["rrf_k"],
                "sparse_weight": contract["weights"]["bm25"],
                "dense_components": [
                    {"name": name, "weight": weight}
                    for name, weight in contract["weights"].items()
                    if name != "bm25"
                ],
                "reference_expansion": {
                    "enabled": True,
                    "scopes": ["same_document"],
                    "seed_top_k": 5,
                    "max_expansions": 4,
                },
                "same_law_neighbor_expansion": {"enabled": False},
                "chunk_candidate_selection": "top_chunks",
            }
        },
        "reranking": {
            "candidate_pool_size": pool_size,
            "retrieved_chunk_depth": 100,
            "candidate_source": "retrieved_chunks",
        },
    }


def test_handoff_contract_is_verified_from_config(tmp_path: Path) -> None:
    handoff = tmp_path / "retrieval_stage_handoff.json"
    handoff.write_text("{}", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_config(50), sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "experiment_name": "frozen_dev_handoff_pool50_refon_articlemilv1",
        "dev_queries": 438,
        "handoff_file": handoff.name,
        "handoff_sha256": hashlib.sha256(handoff.read_bytes()).hexdigest(),
        "config_file": "config.yaml",
        "retrieval_contract": expected_contract(50),
    }
    (tmp_path / "retrieval_handoff_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    config, resolved_handoff, _ = load_and_validate_handoff(tmp_path, 50)

    assert config["reranking"]["candidate_pool_size"] == 50
    assert resolved_handoff == handoff


def test_checkpoint_cases_preserve_explicit_epoch_order(tmp_path: Path) -> None:
    warm_start = tmp_path / "warm"
    checkpoint_root = tmp_path / "checkpoints"
    paths = [warm_start] + [checkpoint_root / f"checkpoint-{step}" for step in (10, 20, 30, 40, 50)]
    for path in paths:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    cases = build_cases(warm_start, checkpoint_root, [10, 20, 30, 40, 50])

    assert [case.label for case in cases] == [
        "warm_start_vietnamese",
        "adaptation_epoch_1",
        "adaptation_epoch_2",
        "adaptation_epoch_3",
        "adaptation_epoch_4",
        "adaptation_epoch_5",
    ]
    assert cases[-1].model_path.name == "checkpoint-50"


def test_selected_checkpoint_reuses_only_complete_materialized_run(
    tmp_path: Path,
) -> None:
    model = tmp_path / "checkpoint-1208"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    payload = handoff / "retrieval_stage_handoff.json"
    payload.write_text("{}", encoding="utf-8")
    (handoff / "config.yaml").write_text(
        yaml.safe_dump(_config(50), sort_keys=False), encoding="utf-8"
    )
    (handoff / "retrieval_handoff_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "experiment_name": "frozen_dev_handoff_pool50_refon_articlemilv1",
                "dev_queries": 438,
                "handoff_file": payload.name,
                "handoff_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "config_file": "config.yaml",
                "retrieval_contract": expected_contract(50),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    run_dir = output / "pointwise_mil_pool50_selected"
    run_dir.mkdir(parents=True)
    for name in REQUIRED_RESULTS:
        (run_dir / name).write_text("{}", encoding="utf-8")

    resolved = evaluate_selected_checkpoint(
        pool_size=50,
        model_path=model,
        label="selected",
        handoff_root=handoff,
        corpus_path=tmp_path / "corpus.parquet",
        train_path=tmp_path / "train.parquet",
        output_root=output,
    )

    assert resolved == run_dir
