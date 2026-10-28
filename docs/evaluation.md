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
