# Kaggle notebooks

Notebooks are grouped by research stage. Paths describe responsibility, not a
separate implementation: reusable pipeline logic remains in `src/drill_ir/`
and `scripts/`.

## Current workflow

### `preparation/`

- `kaggle_build_retrieval_cache.ipynb`: build and verify exactly one sparse,
  dense, or shared-chunk cache artifact per process.

### `experiments/retrieval/`

- `kaggle_table2_chunking_ablation.ipynb`: controlled four-case dense chunking
  comparison for report Table R2.
- `kaggle_table3_first_stage_ablation.ipynb`: controlled six-case first-stage
  component comparison for report Table R3, recomputed offline from the frozen
  fixed-window/reference-off component rankings.
- `kaggle_table4_structure_reference_factorial.ipynb`: controlled 2×2
  fixed-window/Khoản–Điểm × same-document-reference experiment for report
  Table R4, including checkpoint-1208 reranking and per-cell `enhanced_v2`
  LightGBM evaluation.

### `experiments/reranking/`

- `kaggle_evaluate_fixed448_pool50_3epochs.ipynb`: sequentially evaluates the
  Vietnamese warm start and fixed-window/reference-off pointwise-MIL checkpoints
  `1234/2471/3709` on one seed-42 development retrieval handoff. It first looks
  for a compatible handoff retained by the fixed-window epoch-1 submission
  output; otherwise it reuses exact fixed-window caches or rebuilds retrieval
  once. For every model it stores the reranker artifacts and enhanced-v2
  LightGBM feature rows, OOF scores/rankings, fold assignments, metrics, and
  all-dev refit model, then bundles the complete reusable stages.
- `kaggle_pool50_epoch1_lightgbm_feature_screen.ipynb`: historical
  checkpoint-1208 development feature screen; its feature comparison remains
  useful context but is not the current submission entry point.

### `experiments/decision_policy/`

- `kaggle_final_decision_policy_screen.ipynb`: compare nested raw/calibrated
  thresholds, a cardinality model, fixed-top-k controls, score-gap selection,
  and the oracle prefix from saved OOF rankings.

### `submission/`

- `kaggle_pool50_epoch1_lightgbm_submission.ipynb`: historical public/private
  inference for pool 50, checkpoint 1208, `enhanced_v2`, and score-gap policy,
  with fixed-top-2 and threshold controls.
- `kaggle_pool50_epoch4_lightgbm_submission.ipynb` and
  `kaggle_pool50_epoch5_lightgbm_submission.ipynb`: matched checkpoint controls
  for checkpoints 4842 and 6054. Each one regenerates its own development
  reranker artifacts from the frozen pool-50 handoff and refits LightGBM; they
  never reuse epoch-1 reranker scores or a fitted epoch-1 LightGBM model.
- `kaggle_pool50_epoch1_lightgbm_stopping_submission.ipynb`: historical
  checkpoint-1208 experimental
  epoch-1 variant that preserves the frozen ranking pipeline and replaces only
  final score-gap selection with a query-grouped, nested fold-safe learned
  prefix stopping policy. It retains score-gap, threshold, and top-2 outputs as
  controls and packages stopping provenance separately.
- `kaggle_pool50_epoch1_lightgbm_prefix_utility_submission.ipynb`: lightweight
  confirmatory cardinality experiment that reuses the completed stopping run's
  frozen `enhanced_v2` rankings, scores prefixes `k=1..5` with a fold-safe Ridge
  utility model, and produces separate public/private submission ZIPs.
- `kaggle_cardinality_policy_nested_screen.ipynb`: closed five-family policy
  screen on the same frozen rankings. It performs inner hyperparameter selection
  inside every outer query fold, records the nested-OOF winner, and packages a
  public/private ZIP pair for every case as exploratory artifacts.
- `kaggle_cardinality_multiclass_confirmatory.ipynb`: focused follow-up to the
  nested winner. It retains the winning balanced multiclass logistic baseline
  and compares cost-sensitive expected-F2 decoding with an inner-tuned
  multiclass–Ridge ensemble, preserving complete models and provenance.
- `kaggle_final_evidence_subset_selection.ipynb`: final, lightweight decision-
  layer experiment over the same frozen `enhanced_v2` top-five rankings. It
  compares cardinality-plus-binary candidate selection with joint utility over
  all 31 nonempty top-five subsets, reports prefix/subset oracle headroom, saves
  nested-OOF predictions and packages every public/private case. It does not
  rerun retrieval, reranking, or LightGBM.
- `kaggle_pool50_epoch1_fixed_refoff_lightgbm_submission.ipynb`: matched
  fixed 448/64/reference-off end-to-end runner. Its default R3-retuned profile
  uses BM25/Harrier/E5 `0.15/0.65/0.20`; a historical profile preserves
  `0.15/0.55/0.30`. It reuses compatible retrieval caches, reruns affected
  downstream stages, refits `enhanced_v2`, and produces confirmatory
  multiclass–Ridge submissions in the same reproducibility bundle.
- `kaggle_pool50_fixed448_refoff_epoch1_ckpt1234_multiclass_ridge_submission.ipynb`:
  current practical submission runner for the fixed/reference-off pipeline,
  using the owner-designated private-priority multiclass–Ridge baseline. It
  resolves and validates epoch-1 checkpoint `1234` from the attached
  fixed-window three-epoch training notebook output, refits development,
  public/private reranking, `enhanced_v2`, and multiclass–Ridge policy stages,
  and packages the full reusable stage outputs and provenance. It does not
  modify the earlier checkpoint-1208 submission notebook.

### `demo/`

- `kaggle_streamlit_legal_qa.ipynb`: installs and launches the Streamlit UI,
  reads OpenAI and Ngrok credentials from separate Kaggle Secrets, and exposes
  port 8501. It attaches the legal corpus, Harrier/E5 models, checkpoint-1234
  training output, and the current submission's complete-results archive;
  `demo/streamlit/pipeline_adapter.py` serves the frozen retrieval, reranking,
  LightGBM, and Multiclass–Ridge pipeline per question.

## Non-primary notebooks

### `training/deferred/`

Pairwise pool-50 and pool-100 launchers are preserved here because their
multi-GPU training path stalled and is not part of the selected pointwise
pipeline. Do not use them as final-training entry points.

### `diagnostics/`

Runtime/debug variants for reranker batching and the no-reference reranker are
kept separate from reportable experiments. Their outputs are diagnostic, not
main-table evidence.

### `temp/`

Incomplete or transitional launchers. Nothing under this directory is a
supported Run-All entry point.

### `archive/`

Completed, superseded, failed, or deferred historical work, grouped by stage.
See `archive/README.md`. Archived notebooks are retained for provenance but are
not recommended entry points.

## Path policy

- Link to the categorized path in documentation and experiment notes.
- Moving a notebook must not change its scientific configuration.
- Kaggle outputs, caches, checkpoints, and raw data do not belong here.
- A new reportable notebook should go under the stage it evaluates; a thin
  preparation utility should go under `preparation/`.
