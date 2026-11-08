## Latest owner-supplied fixed-top-2 checkpoint results (2026-11-06)

The owner replaced the threshold diagnostic with the following LightGBM
enhanced_v2 fixed-top-2 results. Report Figure 2 now fixes output size at two
for both curves. These results supersede the threshold comparison as the
report's explanation of the chosen checkpoint.

| Checkpoint | Precision@2 | Recall@2 | F2@2 |
|---|---:|---:|---:|
| Warm start | 0.408676 | 0.682725 | 0.601989 |
| Epoch 1 | 0.426941 | 0.701696 | 0.621680 |
| Epoch 2 | 0.423516 | 0.699413 | 0.618791 |
| Epoch 3 | 0.413242 | 0.691857 | 0.609650 |

Report and slide enhanced_v2 rows, fixed-top-2 controls, summary milestones,
and derived differences are updated to 0.4269/0.7017/0.6217. Enhanced_v2 now
leads precision as well as recall/F2 in the displayed schema table. The curve
supports epoch 1 specifically after LightGBM at fixed top-2, while epoch 3
leads reranker-only top-2. No models were run for this correction.

These owner-supplied checkpoint-screen values are not a recomputation of the
newly uploaded bundle's OOF ranking (top-2 F2 0.6113609814). Preserve both
sources. The per-query error audit and ensemble F2 0.6910165043 still use
the uploaded bundle. The display does not establish that all policy rows
share an identical saved ranking; report/slide captions avoid that claim.

# Evaluation


## Saved-run error audit (2026-11-06)

The owner uploaded the current complete-results archive for fixed windows
448/64, reference-off, WRRF10 weights 0.15/0.65/0.20, pool 50, checkpoint
1234 and the multiclass–Ridge policy. Its SHA-256 is
`abceb131eb131e92714376b49a284f147d3e185ce62790e079f97f4caf9c731f`.
The local train-Parquet SHA-256 matches the archived input exactly. All
438 persisted dev queries and 593 deduplicated relevance assignments align.

The archive saves outer-fold metrics and lambda choices, but not per-query
nested-OOF final decisions. `scripts/audit_report_errors.py` reconstructs them
by fitting only the five small outer-fold logistic/Ridge heads using the
saved inner-selected lambdas (0.75 for folds 0–3, 0.5 for fold 4). No neural
model, LightGBM model, retrieval index or hyperparameter search is rerun.
Every outer-fold P/R/F2 and the aggregate reproduce within 1e-12:
0.5757229832572298 / 0.7274353120243532 / 0.6910165043213872.

Measured gold-assignment partition (each (qid, gold aid) counted once):

| Observed state | Assignments |
|---|---:|
| Returned by the final policy | 387 |
| Absent from the actual 50-chunk pool | 60 |
| In pool, below LightGBM top-5 and below reranker top-5 | 84 |
| In pool, reranker top-5 but demoted outside LightGBM top-5 | 13 |
| In LightGBM top-5 but excluded by the selected prefix | 49 |
| Total | 593 |

The report groups the middle ranking conditions into 97 outside top-5, giving
three disjoint missing-gold groups 60/97/49 (206 total). These describe saved
rank positions; they are not controlled causal effects of removing a stage.
Gold-pair counts are not macro-recall. Actual pool macro-recall is
0.9318656229615133; reranker/LightGBM top-5 macro-recall is
0.7795879539030225/0.8012774516199174.

Query-level outcomes: exact gold set 152; gold plus extras 125; only missing
35; both missing and extra 126. Final false-positive assignments total 560;
false-negative assignments total 206. Two illustrative report cases are
qid 14663 (gold 29868/29872; selected 29872; missing gold at LightGBM rank 2)
and qid 2406 (gold 19739; selected 19739/57446). Source text confirms that
57446 concerns fare exemption/reduction, whereas the question asks priority
in buying a ticket. It is described as outside benchmark gold, not as an
authoritative claim that it is legally unrelated in every setting.

**Unresolved cross-source difference:** this archive's actual enhanced_v2
OOF top-2 P/R/F2 is 0.4189497716894977 / 0.690661013263753 /
0.6113609813641646, unlike the previously owner-confirmed slide table
0.4223/0.7027/0.6203. This does not invalidate the arithmetic correction
below: it identifies a different stored ranking from the reported table.
The owner was asked whether a separate newer feature-screen artifact exists.
The report/slide feature-screen values have not been silently overwritten;
error analysis uses the newly uploaded predictions and matches its 0.6910
ensemble score. Confirm the table source before claiming both rows come
from an identical saved ranking.

Audit outputs, per-query ranks, reconstructed selections and case source text
are in ignored `results/report_error_audit_1234/`. The measured aggregates and
input provenance are recorded here so they are not confined to ignored files.

## Resolved metric transcription error (2026-11-05)

The final slide audit found that the previously owner-reported enhanced_v2
fixed-top-2 triplet 0.4223/0.7027/0.6270 was inconsistent with the official
formula. The owner confirmed a transcription error and authorized correction:
Precision **0.4223**, Recall **0.7027**, Macro-F2 **0.6203**. The corrected F2
is recomputed from the reported P/R:
`5*0.4223*0.7027/(4*0.4223+0.7027) = 0.6203231949496216`.

Report tables R6/R7, experiment records, handoff, and slide tables/conclusion
were updated together. This is an owner-confirmed arithmetic correction, not
an artifact-level re-evaluation or new training run. The final audit checked
21 P/R/F2 table rows; the two repeated occurrences of this triplet were the
only discrepancies beyond four-decimal rounding tolerance. Preserve the
owner-reported provenance of all other values.

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

The broader confirmatory screen is frozen in
`configs/cardinality_policy_screen.yaml`: tuned Ridge utility, histogram
gradient-boosting utility, multinomial logistic cardinality, histogram
gradient-boosting cardinality, and pairwise logistic prefix preference. Every
outer fold selects that case's hyperparameters using only inner cross-fitted
training queries. `selected_by_nested_oof.json` is the scientific selection;
the generated ZIP for every case is artifact convenience and must not turn
Codabench into a hyperparameter selector.

After the closed screen selected balanced multiclass logistic, one focused
confirmatory comparison is permitted. It keeps `C=0.1`, balanced classes,
Ridge `alpha=1`, and the frozen rankings fixed. Cost-sensitive decoding chooses
the prefix with maximum expected training-fold utility under multiclass class
probabilities. The ensemble combines standardized log class probabilities with
Ridge utilities; its mixing weight is selected only in inner folds from
`{0.25,0.5,0.75}`. The unchanged multiclass winner is included as the baseline.

The final evidence-set diagnostic is frozen in
`configs/final_evidence_subset.yaml`. Unlike every prefix policy above, it may
select a non-contiguous subset of the frozen top five. It compares exactly two
cases: predicted cardinality plus a binary per-candidate selector, and Ridge
utility over all 31 nonempty top-five subsets. Both retain the same outer/inner
query folds and select hyperparameters inside training folds only. Report both
oracle-prefix and oracle-subset scores: their difference isolates headroom from
non-prefix selection and neither is deployable. This is a final offline
diagnostic, not permission for another open-ended leaderboard search.

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
