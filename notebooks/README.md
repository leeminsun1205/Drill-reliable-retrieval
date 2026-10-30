# Kaggle notebooks

Only current execution entry points live in this directory:

- `kaggle_main_frozen_first_stage.ipynb`: primary post-selection 2-by-2 gate.
  It freezes BM25 + Harrier + multilingual-E5-large, WRRF `k=10`, weights
  `0.15/0.55/0.30`, component depth 100, and the epoch-3 reranker. Change only
  `CANDIDATE_BUDGET` (`50` or `100`) and
  `ENABLE_REFERENCE_EXPANSION` (`False` or `True`).
- `kaggle_finetune_reranker_warmstart_bce.ipynb`: conservative reranker
  adaptation. It warm-starts from the pinned Vietnamese reranker, consumes
  the separately prepared FPT-inspired mining artifact, and reuses a frozen
  development retrieval handoff. It trains pointwise BCE for
  exactly five new adaptation epochs, and evaluates every snapshot.
- `kaggle_finetune_reranker_warmstart_pairwise.ipynb`: promotion-oriented
  treatment using the same audited candidates and Vietnamese warm start, but
  expanding positive-negative pairs and optimizing pairwise logistic loss.
- `kaggle_build_retrieval_cache.ipynb`: build one verified sparse or dense
  cache artifact per process.
- `kaggle_multi_dense_weight_selection.ipynb`: current BM25S + Harrier +
  multilingual-E5 peer-fusion experiment and downstream evaluation.
- `kaggle_multi_dense_harrier_bge.ipynb`: controlled BM25S + Harrier + BGE-M3
  peer-fusion treatment. It keeps the current multi-dense split, WRRF screen,
  reranker, and LightGBM stages, while disabling neighbor and reference
  expansion so that the dense-peer replacement is the intended change.
- `kaggle_multi_dense_harrier_vietlegal_e5.ipynb`: matched no-expansion
  BM25S + Harrier + VietLegal-E5 treatment using the same WRRF screen and
  downstream stages.
- `kaggle_multi_dense_harrier_e5_retuned.ipynb`: no-expansion Harrier +
  multilingual-E5-large control using the expanded WRRF grid. Compare it with
  the reference-enabled main notebook only after confirming all non-reference
  settings and split IDs match.
- `kaggle_query_recall50_analysis.ipynb`: analysis-only view of a completed
  multi-dense archive, including per-query deficit, component attribution,
  stage coverage, and independently downloadable diagnostic tables.
- `kaggle_epoch3_lightgbm_submission.ipynb`: reproduce the selected
  single-dense epoch-3 reranker, LightGBM, and submission pipeline.
- `kaggle_reranker_batch_benchmark.ipynb`: minimal reranker-only, load-once
  benchmark of neural batch sizes 8, 64, and 128 with adaptive repetitions and
  full-run timing extrapolation.
- `kaggle_reranker_stage_smoke.ipynb`: diagnostic full-retrieval run followed
  by a process handoff and a 20-query reranker subset; its metrics are not
  research results and must not feed LightGBM.

Completed, superseded, failed, or deferred experiment launchers are under
`archive/`. They remain available for reproducibility but are not recommended
Run-All entry points. Reusable pipeline logic belongs in `src/drill_ir/` and
`scripts/`, not only in notebooks.
