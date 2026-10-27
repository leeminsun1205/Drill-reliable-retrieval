# Kaggle notebooks

Only current execution entry points live in this directory:

- `kaggle_build_retrieval_cache.ipynb`: build one verified sparse or dense
  cache artifact per process.
- `kaggle_multi_dense_weight_selection.ipynb`: current BM25S + Harrier +
  multilingual-E5 peer-fusion experiment and downstream evaluation.
- `kaggle_epoch3_lightgbm_submission.ipynb`: reproduce the selected
  single-dense epoch-3 reranker, LightGBM, and submission pipeline.
- `kaggle_reranker_batch_benchmark.ipynb`: minimal reranker-only, load-once
  benchmark of neural batch sizes 8 and 16 with adaptive repetitions and
  full-run timing extrapolation.

Completed, superseded, failed, or deferred experiment launchers are under
`archive/`. They remain available for reproducibility but are not recommended
Run-All entry points. Reusable pipeline logic belongs in `src/drill_ir/` and
`scripts/`, not only in notebooks.
