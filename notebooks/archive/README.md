# Archived notebooks

These notebooks preserve completed or superseded experiments for
reproducibility. They are not the current Run-All entry points.

- `kaggle_overnight_experiments.ipynb`: initial broad baseline suite, superseded
  by the focused Khoản–Điểm suite.
- `kaggle_finetune_reranker_448.ipynb`: first Điểm 448/64 fine-tuning treatment,
  superseded by the aligned direct retrieved-chunk BCE treatment.
- `kaggle_finetune_khoan_diem_dual_candidate.ipynb`: completed comparison that
  established direct retrieved-chunk reranking as the selected candidate flow.
- `kaggle_smoke_finetune_reranker.ipynb`: one-step single-GPU and two-GPU DDP
  plumbing checks for reranker training; it is not a reportable experiment.
- `kaggle_rm3_rocchio_ablation_initial.ipynb`: superseded first RM3/Rocchio
  ablation launcher, retained to explain the transition to the current focused
  feedback notebook.

Current notebooks remain one directory above. Do not delete archived notebooks
until their configs, metrics, and limitations are fully represented in the
Markdown sources of truth.

Additional completed or deferred notebooks moved here during the 2026-10-27
cleanup:

- `kaggle_bm25_native_vs_bm25s.ipynb`: reproducibility gate that established
  identical standard-BM25 top-100 rankings and selected bm25s for execution.
- `kaggle_dense_standalone_screening.ipynb`: completed seven-model dense screen;
  aggregate results are recorded in `docs/experiments.md`.
- `kaggle_sparse_bm25_plus_screening.ipynb`: completed BM25 parameter and BM25+
  control screen; standard BM25 1.2/0.75 remains selected.
- `kaggle_khoan_diem_experiments.ipynb`: completed structural representation
  suite that selected direct Khoản–Điểm retrieved-chunk reranking.
- `kaggle_finetune_fpt_article_bce.ipynb`: full-article FPT-style training
  control, superseded for the main path by aligned direct-chunk supervision.
- `kaggle_finetune_main_retrieved_chunks_bce.ipynb`: completed two-epoch aligned
  training run that selected epoch 2 at that stage.
- `kaggle_finetune_main_retrieved_chunks_bce_epochs3_4.ipynb`: completed longer
  training follow-up that selected epoch 3.
- `kaggle_rm3_rocchio_ablation.ipynb`: deferred feedback screen after exceeding
  the practical Kaggle runtime budget; feedback remains disabled.

The three current entry points and their roles are listed in
`notebooks/README.md`.
