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
  exactly five new adaptation epochs with the next seed-determined slice of at
  most four article chunks per epoch, and evaluates every snapshot over all
  frozen candidate chunks.
- `kaggle_finetune_reranker_warmstart_bce_pool100.ipynb`: matched pointwise-BCE
  cost/ceiling treatment. It keeps the pool-50 training and evaluation protocol
  fixed and changes only the candidate budget and corresponding artifact IDs.
- `kaggle_evaluate_pointwise_mil_pool50_checkpoints.ipynb`: evaluation-only
  recovery/selection entry point for the completed pool-50 run. It reads the
  five checkpoints from the attached fine-tuning notebook output and reuses the
  verified pool-50 handoff; it does not load retrieval or mining models.
- `kaggle_evaluate_pointwise_mil_pool100_checkpoints.ipynb`: matched
  evaluation-only entry point for the five completed pool-100 checkpoints.
  Both evaluation notebooks run warm start and epochs 1--5 in separate,
  sequential subprocesses so model and CUDA state are released between cases.
- `kaggle_pool50_epoch1_lightgbm_feature_screen.ipynb`: post-selection entry
  point for checkpoint 1208. It reuses the verified pool-50 handoff, creates
  complete enriched development artifacts once, precomputes corpus-static
  article features, and compares the 15/41/61/69-feature LightGBM schemas with
  identical query-grouped OOF folds and selection policies.
- `kaggle_pool50_epoch1_lightgbm_submission.ipynb`: final Run-All inference
  entry point for the selected pool-50 epoch-1 checkpoint `1208`. It consumes
  the retained feature-screen dev artifact, runs the frozen reference-on
  public/private retrieval and reranker once, deterministically rebuilds and
  checksum-verifies graph v3 from the corpus, fits `enhanced_v2` on all labeled
  development rows, and packages fixed-top-2, threshold/fallback-2, and the
  selected score-gap `{3,4,5}` policies for both official splits.
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

The pairwise pool-50 and pool-100 notebooks remain at their historical paths
for auditability, but are deferred after repeated multi-GPU stalls and are not
part of the current execution workflow. The active reranker comparison is
pointwise BCE pool 50 versus pointwise BCE pool 100.
