"""Interactive adapter for the frozen pool-50 fixed-window DRiLL pipeline.

The service loads retrieval indexes once, then reuses repository retrieval,
reranking, LightGBM feature generation, and cardinality-selection code for each
question. It consumes the complete-results archive from the selected Kaggle
submission notebook; it does not train or tune models at demo startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
from typing import Any

import numpy as np

from drill_ir.analysis.cardinality_confirmatory import apply_confirmatory_case
from drill_ir.analysis.ltr import build_ltr_examples
from drill_ir.data import iter_articles
from drill_ir.data.models import Query
from drill_ir.reranking import build_reranker
from drill_ir.retrieval import build_retriever
from drill_ir.text_features import chunk_pair_features

from backend import Evidence


ARTIFACT_GLOB = "*pool50_fixed448_refoff_epoch1_ckpt1234*complete_results.tar.gz"
WORK_ROOT = Path("/kaggle/working/drill_streamlit_demo")


def _unique_file(pattern: str, *, root: Path = Path("/kaggle/input")) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one input matching {pattern!r}; found {matches}")
    return matches[0]


def _model_root(slug: str) -> Path:
    matches = []
    for config in Path("/kaggle/input").rglob("config.json"):
        root = config.parent
        if slug not in str(root).lower():
            continue
        if any(root.glob("*.safetensors")) or (root / "pytorch_model.bin").is_file():
            matches.append(root)
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one attached model for {slug!r}; found {matches}")
    return matches[0]


def _checkpoint_root() -> Path:
    matches = []
    for manifest_path in Path("/kaggle/input").rglob("checkpoint_manifest.json"):
        checkpoint = manifest_path.parent
        if checkpoint.name != "checkpoint-1234":
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("objective") == "pointwise_mil_bce"
            and manifest.get("epoch_completed") == 1
            and manifest.get("global_step") == 1234
            and (checkpoint / "config.json").is_file()
            and any((checkpoint / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
        ):
            matches.append(checkpoint)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one verified fixed-window epoch-1 checkpoint-1234; found {matches}")
    return matches[0]


def _safe_extract(archive_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        root = target.resolve()
        for member in archive.getmembers():
            resolved = (target / member.name).resolve()
            if root not in resolved.parents and resolved != root:
                raise ValueError(f"Unsafe path in results archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        archive.extractall(target, filter="data")


def _resolve_retrieval_cache(slug: str) -> tuple[Path, bool]:
    """Resolve one verified attached cache, or allocate a working cache path."""
    input_root = Path("/kaggle/input")
    manifests = sorted(input_root.rglob(f"manifest_{slug}.json"))
    archives = sorted(input_root.rglob(f"retrieval_cache_{slug}.tar"))
    if len(manifests) > 1 or (not manifests and len(archives) > 1):
        raise RuntimeError(f"Ambiguous attached retrieval cache {slug}: manifests={manifests}, archives={archives}")
    if manifests:
        manifest = manifests[0]
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        root = manifest.parent / slug
        if payload.get("selection", {}).get("slug") != slug or not payload.get("verified_cache_hit") or not root.is_dir():
            raise RuntimeError(f"Invalid attached retrieval cache manifest: {manifest}")
        return root, True
    if archives:
        target = WORK_ROOT / "attached_caches" / slug
        manifest = target / "retrieval-cache" / f"manifest_{slug}.json"
        if not manifest.is_file():
            _safe_extract(archives[0], target)
        if not manifest.is_file():
            raise RuntimeError(f"Cache archive is missing its manifest: {archives[0]}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        root = manifest.parent / slug
        if payload.get("selection", {}).get("slug") != slug or not payload.get("verified_cache_hit") or not root.is_dir():
            raise RuntimeError(f"Invalid retrieval cache archive: {archives[0]}")
        return root, True
    expanded = sorted(path for path in input_root.rglob(slug) if path.is_dir())
    if len(expanded) > 1:
        raise RuntimeError(f"Ambiguous expanded retrieval cache {slug}: {expanded}")
    if expanded:
        return expanded[0], True
    return WORK_ROOT / "caches" / slug, False


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint(archive_root: Path, checkpoint: Path) -> None:
    provenance_path = archive_root / "checkpoint_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("The complete-results archive lacks checkpoint_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest is missing: {manifest_path}")
    expected_manifest = provenance.get("checkpoint_manifest_sha256")
    if expected_manifest != _sha256(manifest_path):
        raise RuntimeError("Attached checkpoint-1234 manifest differs from the selected submission artifact")
    for name, expected in provenance.get("model_files", {}).items():
        path = checkpoint / name
        if not path.is_file() or _sha256(path) != expected.get("sha256"):
            raise RuntimeError(f"Attached checkpoint file does not match the selected artifact: {path}")


def _restore_cardinality_model(model_path: Path) -> dict[str, Any]:
    """Restore the exact fitted sklearn objects persisted by the submitter."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    payload = json.loads(model_path.read_text(encoding="utf-8"))

    def scaler_from(value: dict[str, Any]) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(value["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(value["scale"], dtype=np.float64)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.n_samples_seen_ = 1
        return scaler

    query_scaler = scaler_from(payload["query_scaler"])
    utility_scaler = scaler_from(payload["utility_scaler"])

    multiclass_payload = payload["multiclass"]
    multiclass = LogisticRegression()
    multiclass.classes_ = np.asarray(multiclass_payload["classes"], dtype=np.int64)
    multiclass.coef_ = np.asarray(multiclass_payload["coefficient"], dtype=np.float64)
    multiclass.intercept_ = np.asarray(multiclass_payload["intercept"], dtype=np.float64)
    multiclass.n_features_in_ = multiclass.coef_.shape[1]
    multiclass.n_iter_ = np.asarray([1], dtype=np.int32)

    ridge_payload = payload["ridge"]
    ridge = Ridge()
    ridge.coef_ = np.asarray(ridge_payload["coefficient"], dtype=np.float64)
    ridge.intercept_ = float(ridge_payload["intercept"])
    ridge.n_features_in_ = ridge.coef_.shape[-1]

    return {
        "query_scaler": query_scaler,
        "multiclass": multiclass,
        "utility_scaler": utility_scaler,
        "ridge": ridge,
        "utility_matrix": np.asarray(payload["utility_matrix"], dtype=np.float64),
    }


@dataclass
class _Artifacts:
    config: dict[str, Any]
    lightgbm_model: Any
    lightgbm_feature_names: tuple[str, ...]
    cardinality_model: dict[str, Any]
    cardinality_deployment: dict[str, Any]
    maximum_k: int


def _load_artifacts(archive_path: Path, extraction: Path, checkpoint: Path) -> _Artifacts:
    import lightgbm
    import yaml

    if not extraction.exists():
        _safe_extract(archive_path, extraction)
    _verify_checkpoint(extraction, checkpoint)

    dev_configs = []
    for split_path in extraction.rglob("split_dev_qids.json"):
        candidate = split_path.parent / "config.yaml"
        if candidate.is_file() and "dev_run" in candidate.parts:
            dev_configs.append(candidate)
    if len(dev_configs) != 1:
        raise RuntimeError(f"Could not resolve one development config from {archive_path}: {dev_configs}")
    config = yaml.safe_load(dev_configs[0].read_text(encoding="utf-8"))
    retrieval = config["retrieval"]
    hybrid = retrieval["hybrid"]
    chunking = retrieval["dense"]["chunking"]
    components = {row["name"]: row for row in hybrid["dense_components"]}
    if (
        config.get("method") != "hybrid_reranker"
        or int(config.get("seed", -1)) != 42
        or int(retrieval.get("top_k", -1)) != 100
        or int(hybrid.get("rrf_k", -1)) != 10
        or abs(float(hybrid.get("sparse_weight", -1)) - 0.15) > 1e-9
        or hybrid.get("reference_expansion", {}).get("enabled") is not False
        or hybrid.get("same_law_neighbor_expansion", {}).get("enabled") is not False
        or chunking.get("method") != "token_window"
        or int(chunking.get("max_tokens", -1)) != 448
        or int(chunking.get("overlap_tokens", -1)) != 64
        or int(config.get("reranking", {}).get("candidate_pool_size", -1)) != 50
        or int(config.get("reranking", {}).get("retrieved_chunk_depth", -1)) != 100
        or set(components) != {"vietlegal_harrier", "multilingual_e5_large"}
        or abs(float(components["vietlegal_harrier"].get("weight", -1)) - 0.65) > 1e-9
        or abs(float(components["multilingual_e5_large"].get("weight", -1)) - 0.20) > 1e-9
    ):
        raise RuntimeError("Complete-results archive does not match the frozen fixed-window pool-50 baseline")

    lgbm_candidates = list(extraction.rglob("lightgbm_enhanced_v2/lightgbm_final_model.txt"))
    card_candidates = list(extraction.rglob("multiclass_ridge_and_cardinality/multiclass_ridge_ensemble/model.json"))
    deployment_candidates = list(extraction.rglob("multiclass_ridge_and_cardinality/multiclass_ridge_ensemble/deployment_selection.json"))
    provenance_candidates = list(extraction.rglob("lightgbm_enhanced_v2/provenance.json"))
    groups = (lgbm_candidates, card_candidates, deployment_candidates, provenance_candidates)
    if any(len(group) != 1 for group in groups):
        raise RuntimeError(f"Incomplete or ambiguous model artifacts in {archive_path}: {[list(map(str, g)) for g in groups]}")

    provenance = json.loads(provenance_candidates[0].read_text(encoding="utf-8"))
    if provenance.get("feature_schema") != "enhanced_v2" or len(provenance.get("feature_names", [])) != 61:
        raise RuntimeError("LightGBM artifact is not the frozen 61-feature enhanced_v2 model")
    card_directory = card_candidates[0].parent
    card_provenance = json.loads((card_directory / "provenance.json").read_text(encoding="utf-8"))
    if card_provenance.get("case") != "multiclass_ridge_ensemble":
        raise RuntimeError("Cardinality artifact does not contain the selected Multiclass–Ridge ensemble")
    maximum_k = int(card_provenance["config"]["maximum_k"])
    if maximum_k != 5:
        raise RuntimeError(f"Expected final cardinality maximum_k=5; got {maximum_k}")

    return _Artifacts(
        config=config,
        lightgbm_model=lightgbm.Booster(model_file=str(lgbm_candidates[0])),
        lightgbm_feature_names=tuple(provenance["feature_names"]),
        cardinality_model=_restore_cardinality_model(card_candidates[0]),
        cardinality_deployment=json.loads(deployment_candidates[0].read_text(encoding="utf-8")),
        maximum_k=maximum_k,
    )


class FixedWindowPool50Backend:
    """Serve one query with the frozen first-stage and decision pipeline."""

    def __init__(self) -> None:
        corpus_path = Path(os.environ.get("DRILL_CORPUS_PATH") or _unique_file("legal-corpus.parquet"))
        archive_path = Path(os.environ.get("DRILL_COMPLETE_RESULTS_ARCHIVE") or _unique_file(ARTIFACT_GLOB))
        checkpoint = Path(os.environ.get("DRILL_CHECKPOINT_PATH") or _checkpoint_root())
        harrier = Path(os.environ.get("DRILL_HARRIER_PATH") or _model_root("vietlegal-harrier"))
        e5 = Path(os.environ.get("DRILL_E5_PATH") or _model_root("multilingual-e5-large"))
        jar = Path(os.environ.get("DRILL_VNCORENLP_JAR", "/kaggle/working/VnCoreNLP/VnCoreNLP-1.2.jar"))
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        if not jar.is_file():
            raise FileNotFoundError(f"VnCoreNLP JAR not found: {jar}")

        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        self.artifact_root = WORK_ROOT / "complete_results"
        artifacts = _load_artifacts(archive_path, self.artifact_root, checkpoint)
        config = artifacts.config
        self._artifacts = artifacts
        self._query_counter = int(time.time()) * 1_000_000

        # Apply only environment paths and safe inference runtime settings. The
        # scientific retrieval/feature/decision parameters come from the archive.
        config["data"]["corpus_path"] = str(corpus_path)
        profiles = config["retrieval"]["dense_profiles"]
        cache_specs = {
            "bm25": "bm25s_vncorenlp",
            "chunks": "bge_m3_token_window448_chunks",
            "harrier": "vietlegal_harrier_0_6b_token_window448",
            "e5": "multilingual_e5_large_token_window448",
        }
        resolved_caches = {name: _resolve_retrieval_cache(slug) for name, slug in cache_specs.items()}
        bm25_cache, bm25_hit = resolved_caches["bm25"]
        config["retrieval"]["bm25"]["cache"].update(
            {"enabled": True, "require_hit": bm25_hit, "directory": str(bm25_cache)}
        )
        chunk_cache, _ = resolved_caches["chunks"]
        config["retrieval"]["dense"]["chunk_cache"].update(
            {"enabled": True, "directory": str(chunk_cache)}
        )
        harrier_cache, harrier_hit = resolved_caches["harrier"]
        e5_cache, e5_hit = resolved_caches["e5"]
        profiles["vietlegal_harrier"]["cache"].update(
            {"enabled": True, "require_hit": harrier_hit, "directory": str(harrier_cache)}
        )
        profiles["multilingual_e5_large"]["cache"].update(
            {"enabled": True, "require_hit": e5_hit, "directory": str(e5_cache)}
        )
        config["retrieval"]["bm25"]["preprocessing"]["tokenizer"].update(
            {"word_segmenter": "vncorenlp", "vncorenlp_jar_path": str(jar)}
        )
        profiles = config["retrieval"]["dense_profiles"]
        profiles["vietlegal_harrier"]["encoder"].update(
            {"model_name": str(harrier), "cache_model_name": "mainguyen9/vietlegal-harrier-0.6b"}
        )
        profiles["multilingual_e5_large"]["encoder"].update(
            {"model_name": str(e5), "cache_model_name": "intfloat/multilingual-e5-large"}
        )
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            gpu_count = 0
        if gpu_count >= 2:
            runtime = {"accelerator": "cuda", "devices": 1, "mixed_precision": False}
            reranker_device = "cuda:1"
        elif gpu_count == 1:
            runtime = {"accelerator": "cuda", "devices": 1, "mixed_precision": False}
            reranker_device = "cuda:0"
        else:
            runtime = {"accelerator": "cpu", "devices": "auto", "mixed_precision": False}
            reranker_device = "cpu"
        config["reranking"].update(
            {
                "model_name": str(checkpoint),
                "batching": "per_query",
                "batch_size": 16,
                "fp16": reranker_device.startswith("cuda:"),
                "device": reranker_device,
            }
        )
        config["runtime"] = runtime

        self.articles = list(iter_articles(corpus_path, include_blank=False))
        self.articles_by_aid = {article.aid: article for article in self.articles}
        if len(self.articles_by_aid) != len(self.articles):
            raise ValueError("Corpus has duplicate article IDs; refusing to serve ambiguous evidence")

        print({
            "demo_corpus": str(corpus_path),
            "complete_results_archive": str(archive_path),
            "checkpoint": str(checkpoint),
            "harrier": str(harrier),
            "e5": str(e5),
            "cache_hits": {name: hit for name, (_path, hit) in resolved_caches.items()},
        }, flush=True)
        print("[demo] Building frozen WRRF first-stage indexes", flush=True)
        retriever = build_retriever("hybrid", config["retrieval"], config["runtime"])
        try:
            retriever.fit(self.articles)
            print("[demo] Loading checkpoint-1234 cross-encoder", flush=True)
            reranker = build_reranker(config["reranking"], config["runtime"])
        except Exception:
            close = getattr(retriever, "close", None)
            if callable(close):
                close()
            raise
        self.retriever = retriever
        self.reranker = reranker
        self.reranker_config = config["reranking"]

    def search(self, question: str) -> list[Evidence]:
        text = str(question).strip()
        if not text:
            return []
        self._query_counter += 1
        query = Query(qid=self._query_counter, question=text, relevant_aids=())
        retrieval_top_k = int(self._artifacts.config["retrieval"]["top_k"])
        first_stage = self.retriever.retrieve([query], top_k=retrieval_top_k)
        component_results = self.retriever.component_results()
        candidates = self.retriever.retrieve_chunks(
            [query],
            top_k=int(self.reranker_config["candidate_pool_size"]),
            dense_depth=int(self.reranker_config["retrieved_chunk_depth"]),
        )
        reranked = self.reranker.rerank_chunks([query], candidates)
        chunk_scores = self.reranker.chunk_scores_by_qid()[query.qid]
        if not reranked or not reranked[0].aids:
            return []

        with tempfile.TemporaryDirectory(prefix="drill-query-") as temporary:
            run_dir = Path(temporary)
            _write_jsonl(run_dir / "predictions.jsonl", [{
                "qid": query.qid,
                "retrieved_aids": list(reranked[0].aids),
                "scores": list(reranked[0].scores),
            }])
            _write_jsonl(run_dir / "first_stage_predictions.jsonl", [{
                "qid": query.qid,
                "retrieved_aids": list(first_stage[0].aids),
                "scores": list(first_stage[0].scores),
            }])
            component_record = {"qid": query.qid, "components": {}}
            for component, results in component_results.items():
                result = results[0]
                component_record["components"][component] = {
                    "retrieved_aids": list(result.aids), "scores": list(result.scores)
                }
            _write_jsonl(run_dir / "component_rankings.jsonl", [component_record])

            chunk_result = candidates[0]
            if len(chunk_scores) > len(chunk_result.chunks):
                raise RuntimeError("Cross-encoder scores do not align with retrieved chunks")
            chunk_records = []
            for chunk, fusion_score, reranker_score in zip(
                chunk_result.chunks[:len(chunk_scores)],
                chunk_result.scores[:len(chunk_scores)],
                chunk_scores,
                strict=True,
            ):
                chunk_records.append({
                    "chunk_id": chunk.chunk_id,
                    "aid": chunk.aid,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count,
                    "fusion_score": fusion_score,
                    "reranker_score": reranker_score,
                    **chunk_pair_features(text, chunk.text),
                })
            _write_jsonl(run_dir / "reranker_chunk_candidates.jsonl", [{"qid": query.qid, "chunks": chunk_records}])

            examples = build_ltr_examples(run_dir, [query], feature_schema="enhanced_v2")
            if tuple(examples["feature_names"]) != self._artifacts.lightgbm_feature_names:
                raise RuntimeError(
                    "Runtime LightGBM feature order differs from the trained model: "
                    f"runtime={len(examples['feature_names'])}, "
                    f"model={len(self._artifacts.lightgbm_feature_names)}"
                )
            lgbm_scores = np.asarray(
                self._artifacts.lightgbm_model.predict(examples["features"]),
                dtype=np.float64,
            )

        order = sorted(
            range(len(examples["aids"])),
            key=lambda index: (-float(lgbm_scores[index]), int(examples["original_ranks"][index])),
        )
        ranked_aids = [int(examples["aids"][index]) for index in order]
        ranked_scores = [float(lgbm_scores[index]) for index in order]
        selected, _diagnostics = apply_confirmatory_case(
            "multiclass_ridge_ensemble",
            self._artifacts.cardinality_model,
            {query.qid: (ranked_aids, ranked_scores)},
            self._artifacts.maximum_k,
            self._artifacts.cardinality_deployment,
        )
        selected_aids = selected[query.qid]
        evidence = []
        score_by_aid = dict(zip(ranked_aids, ranked_scores, strict=True))
        for aid in selected_aids:
            article = self.articles_by_aid.get(int(aid))
            if article is None:
                raise KeyError(f"Selected aid is absent from the indexed corpus: {aid}")
            evidence.append(Evidence(
                aid=str(article.aid),
                text=article.text,
                law_id=article.law_id,
                score=score_by_aid.get(article.aid),
            ))
        return evidence

    def close(self) -> None:
        self.retriever.close()
        self.reranker.close()


def create_backend() -> FixedWindowPool50Backend:
    """Factory consumed by Streamlit's `DRILL_RETRIEVER_FACTORY`."""
    return FixedWindowPool50Backend()
