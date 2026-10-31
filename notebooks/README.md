# Kaggle notebooks

Only current Run-All entry points live in this directory:

- `kaggle_build_retrieval_cache.ipynb`: build and verify one sparse or dense
  retrieval-cache artifact per process.
- `kaggle_pool50_epoch1_lightgbm_feature_screen.ipynb`: reproduce the selected
  checkpoint-1208 development artifact and the controlled 15/41/61/69-feature
  LightGBM screen.
- `kaggle_final_decision_policy_screen.ipynb`: use the saved `enhanced_v2` OOF
  rankings to compare nested raw/calibrated thresholds, a cardinality model,
  fixed top-k controls, and the oracle prefix without reranking or test data.
- `kaggle_pool50_epoch1_lightgbm_submission.ipynb`: final public/private
  inference for pool 50, checkpoint 1208, `enhanced_v2`, and the frozen
  score-gap policy, with fixed-top-2 and threshold controls.

Completed, superseded, failed, diagnostic, and deferred notebooks are retained
under `archive/` for reproducibility. They are not recommended entry points.
Reusable pipeline logic belongs in `src/drill_ir/` and `scripts/`.
