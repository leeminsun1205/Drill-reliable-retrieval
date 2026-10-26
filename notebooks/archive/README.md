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
