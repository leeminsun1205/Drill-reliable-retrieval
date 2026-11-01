# Archived notebooks

These notebooks preserve completed, superseded, failed, or deferred work for
reproducibility. They are not current Run-All entry points. Current notebooks
and their roles are listed in `../README.md`.

## Layout

- `preparation/`: completed mining and frozen-development handoff builders.
- `retrieval/`: sparse/dense screening, fusion, feedback, structural chunking,
  and earlier first-stage suites.
- `training/`: superseded article/chunk BCE and pairwise fine-tuning launchers.
- `evaluation/`: historical checkpoint and frozen-reranker evaluations.
- `submission/`: superseded final-inference launchers.
- `diagnostics/`: smoke tests, batch benchmarks, and recall analysis.

## Notable provenance records

- `retrieval/kaggle_bm25_native_vs_bm25s.ipynb` established identical standard
  BM25 top-100 rankings before selecting bm25s as the execution backend.
- `retrieval/kaggle_dense_standalone_screening.ipynb` contains the completed
  seven-model dense screen.
- `retrieval/kaggle_khoan_diem_experiments.ipynb` records the earlier
  structure-aware representation suite.
- `training/kaggle_finetune_main_retrieved_chunks_bce.ipynb` and its epochs
  3–4 follow-up preserve the earlier aligned BCE training path.
- `retrieval/kaggle_rm3_rocchio_ablation.ipynb` records the deferred feedback
  experiment after it exceeded the practical Kaggle runtime budget.

Do not delete an archived notebook until its configuration, measured results,
and limitations are represented in the Markdown sources of truth. Do not use
archive results as current evidence without checking their split, candidate
set, metric, and configuration against the active protocol.
