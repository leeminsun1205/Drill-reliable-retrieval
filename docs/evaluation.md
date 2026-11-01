# Evaluation

## Split discipline

Avoid evaluation leakage:

- use a fixed random seed for local splitting;
- persist the resulting query IDs;
- reuse exactly the same split across compared methods;
- tune hyperparameters only on train/dev;
- do not select thresholds, top-k, fusion weights, models, prompts, or reranker
  parameters using the final test set.

Treat the official DRiLL private test on Codabench as a final post-challenge
benchmark, not as a development set.

## Current holdout decision

As of 2026-10-28, do not carve an additional local holdout out of the 1,752
queries outside the fixed 438-query seed-42 development split. Those queries
provide scarce reranker supervision, and evaluating the existing epoch-3
reranker on a newly selected subset of them would not be untouched because that
model was trained from the same pool. Creating a valid new holdout would
therefore require excluding it and retraining every supervised downstream
component, with a material reduction in training data.

Continue to use the fixed development split for controlled paired diagnostics,
label all such results as development results, and use query-grouped OOF or
nested-OOF estimates for learned downstream selection. Use official Codabench
evaluation as external confirmation after a treatment and its output policy are
frozen. Merely producing public/private prediction files is not test evidence;
record the official score, split, policy, and submission history before citing
an official improvement. Repeated public-leaderboard submissions used to choose
a method make that leaderboard a tuning signal; never treat it as an untouched
test afterward. Keep the private test final and do not use it to iterate.

## Current pointwise selection protocol

As of 2026-10-30, checkpoint selection and learned final-set selection are two
separate decisions. Compare the pool-50 and pool-100 pointwise MIL snapshots on
the same frozen retrieval handoff and select the neural snapshot by fixed-top-2
development F2. This gate selected pool-50 epoch 1 / checkpoint `1208`. A
threshold tuned on those same 438 queries is diagnostic and must not determine
the checkpoint while also being reported as an unbiased result.

After the checkpoint is frozen, compare LightGBM feature schemas using the same
query-grouped five-fold assignments. Use nested-OOF score-gap F2 as the primary
estimate for the learned output-set policy. Report pooled OOF threshold and
pooled score-gap results separately as tuning/deployment diagnostics. Feature
importance does not validate a feature; feature-group removal is stronger but
remains exploratory when many groups are tried on the same development split.
An official test score is required before claiming external confirmation.

## Nested final-set policy protocol

Once the reranker and LightGBM feature schema are frozen, final-set policies
must be compared on the same saved LightGBM OOF rankings. Assign development
queries deterministically to outer folds. For every outer fold, fit any score
calibrator, tune any threshold, or fit any output-cardinality model using only
the other folds, then predict the held-out queries. Concatenate held-out
predictions before computing the primary macro Precision, Recall, and F2.

The controlled policy screen includes fixed top-k controls, raw-score
thresholding, Platt- and isotonic-calibrated thresholding, and a small
query-cardinality classifier. Report an oracle prefix only as an upper-bound
diagnostic. Pooled-OOF threshold tuning is a deployment diagnostic, not the
unbiased primary estimate. Select a replacement policy only from nested
development results; public and private test scores remain reporting evidence
and must never train or select the policy.

The confirmatory listwise prefix-utility treatment uses those same frozen OOF
rankings, outer query folds, seed, and maximum output size five. It fits a Ridge
regressor with fixed `alpha=10` to the observed per-query F2 of every available
prefix and selects the highest predicted-utility prefix, breaking ties toward
smaller output. The alpha, target, and tie rule are fixed in
`configs/listwise_prefix_utility.yaml`; public/private labels cannot select or
modify them. Compare its concatenated outer-fold metrics directly with binary
stopping before interpreting an official submission.

## Metrics

The primary DRiLL-compatible metrics are Precision, Recall, and F2, aggregated
over queries using the official macro protocol.

For query \(q\), let \(G_q\) be the set of relevant article IDs and \(P_q\) the
set of predicted article IDs:

```text
precision_q = |P_q ∩ G_q| / |P_q|
recall_q    = |P_q ∩ G_q| / |G_q|
```

The task description defines reported Precision and Recall as the arithmetic
mean of these per-query values:

```text
Precision = mean_q(precision_q)
Recall    = mean_q(recall_q)
F2        = (5 * Precision * Recall) / (4 * Precision + Recall)
```

Implement the official formula in that stated order: first macro-average the
per-query precision and recall, then calculate F2 from those two macro values.
Do not substitute `mean_q(F2_q)`, which is generally different.

Diagnostic ranking metrics may include Recall@k, MRR@k, and nDCG@k. Their cutoff
and aggregation conventions must be explicit. Do not silently change metric
definitions between experiments.

The implemented candidate-stage diagnostic is macro Recall@k:

```text
recall_at_k_q = |top_k(q) ∩ G_q| / |G_q|
Recall@k      = mean_q(recall_at_k_q)
```

Cutoffs are declared in the experiment YAML and cannot exceed the retrieved
candidate depth. Missing query rankings contribute zero. Duplicate article IDs
within the first k positions do not increase recall. `Recall@k` measures whether
the first-stage candidate generator preserves gold evidence; it does not replace
official Precision, Recall, or F2 computed on the final selected set.

Metric implementations must be deterministic, independently testable, and
documented. Verify them with hand-computable cases before reporting results:

```text
gold = [2, 5]
pred = [2, 3, 5]
```

For set-based evaluation this example has two true positives, one false
positive, and no false negatives. Unit tests should assert the chosen precision,
recall, zero-prediction behavior, duplicate handling, and F2 aggregation
precisely.

Do not document metric values as measured results unless an executed experiment
artifact supports them.
