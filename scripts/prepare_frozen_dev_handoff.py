#!/usr/bin/env python3
"""Build one frozen development retrieval handoff without loading a reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drill_ir.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, choices=(50, 100), required=True)
    parser.add_argument("--vncorenlp-jar", type=Path, required=True)
    parser.add_argument("--reference-graph", type=Path, required=True)
    parser.add_argument("--bm25-cache", type=Path, required=True)
    parser.add_argument("--chunk-cache", type=Path, required=True)
    parser.add_argument("--harrier-cache", type=Path, required=True)
    parser.add_argument("--e5-cache", type=Path, required=True)
    parser.add_argument("--harrier-model", type=Path, required=True)
    parser.add_argument("--e5-model", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.base_config,
        args.corpus_path,
        args.train_path,
        args.vncorenlp_jar,
        args.reference_graph,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.bm25_cache,
        args.chunk_cache,
        args.harrier_cache,
        args.e5_cache,
        args.harrier_model,
        args.e5_model,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["experiment_name"] = f"frozen_dev_handoff_pool{args.pool_size}_refon_articlemilv1"
    config["method"] = "hybrid_reranker"
    config["data"].update(
        corpus_path=str(args.corpus_path), train_path=str(args.train_path)
    )
    config["output"]["root_dir"] = str(args.output_root)
    config["retrieval"]["top_k"] = 100
    config["retrieval"]["output_k"] = 3

    bm25 = config["retrieval"]["bm25"]
    bm25.update(variant="bm25", delta=1.0)
    bm25["preprocessing"]["tokenizer"].update(
        word_segmenter="vncorenlp", vncorenlp_jar_path=str(args.vncorenlp_jar)
    )
    bm25["cache"].update(
        enabled=True, require_hit=True, directory=str(args.bm25_cache)
    )

    dense = config["retrieval"]["dense"]
    dense["chunk_cache"].update(
        enabled=True, directory=str(args.chunk_cache)
    )
    dense["cache"].update(enabled=True, require_hit=True)
    hybrid = config["retrieval"]["hybrid"]
    hybrid.update(rrf_k=10, sparse_weight=0.15, chunk_candidate_selection="top_chunks")
    hybrid["same_law_neighbor_expansion"]["enabled"] = False
    hybrid["reference_expansion"].update(
        enabled=True,
        graph_path=str(args.reference_graph),
        scopes=["same_document"],
        seed_top_k=5,
        max_expansions=4,
    )
    hybrid["dense_components"] = [
        {"name": "vietlegal_harrier", "profile": "vietlegal_harrier", "weight": 0.55, "top_k": 100},
        {"name": "multilingual_e5_large", "profile": "multilingual_e5_large", "weight": 0.30, "top_k": 100},
    ]
    config["retrieval"]["dense_profiles"] = {
        "vietlegal_harrier": {
            "encoder": {
                "model_name": str(args.harrier_model),
                "cache_model_name": "mainguyen9/vietlegal-harrier-0.6b",
                "query_prefix": "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: ",
                "document_prefix": "",
                "trust_remote_code": True,
            },
            "cache": {"directory": str(args.harrier_cache)},
        },
        "multilingual_e5_large": {
            "encoder": {
                "model_name": str(args.e5_model),
                "cache_model_name": "intfloat/multilingual-e5-large",
                "query_prefix": "query: ",
                "document_prefix": "passage: ",
                "trust_remote_code": False,
            },
            "cache": {"directory": str(args.e5_cache)},
        },
    }
    config["reranking"].update(
        candidate_source="retrieved_chunks",
        candidate_pool_size=args.pool_size,
        retrieved_chunk_depth=100,
    )
    config.setdefault("runtime", {}).update(
        devices=2,
        process_handoff_before_reranking=True,
        stop_after_retrieval_handoff=True,
    )

    resolved = {
        "rrf_k": hybrid["rrf_k"],
        "weights": {
            "bm25": hybrid["sparse_weight"],
            **{item["name"]: item["weight"] for item in hybrid["dense_components"]},
        },
        "candidate_budget": config["reranking"]["candidate_pool_size"],
        "reference_enabled": hybrid["reference_expansion"]["enabled"],
        "reference_scopes": hybrid["reference_expansion"]["scopes"],
        "candidate_source": config["reranking"]["candidate_source"],
        "chunk_candidate_selection": hybrid["chunk_candidate_selection"],
    }
    expected = {
        "rrf_k": 10,
        "weights": {"bm25": 0.15, "vietlegal_harrier": 0.55, "multilingual_e5_large": 0.30},
        "candidate_budget": args.pool_size,
        "reference_enabled": True,
        "reference_scopes": ["same_document"],
        "candidate_source": "retrieved_chunks",
        "chunk_candidate_selection": "top_chunks",
    }
    if resolved != expected:
        raise RuntimeError(f"Frozen handoff config drift: {resolved}")
    print(json.dumps({"frozen_handoff_contract": resolved}, ensure_ascii=False, indent=2), flush=True)
    run_experiment(config)


if __name__ == "__main__":
    main()
