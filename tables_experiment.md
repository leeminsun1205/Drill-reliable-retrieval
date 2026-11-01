# Final Experimental Tables for Report and Slides

## Usage contract

This file contains exactly nine report tables for the 15-page final report.
Five of them are also used in the presentation as reduced views. There are no
slide-only result tables: every number shown on slides must already appear in
the corresponding report table.

Detailed screening grids, full protocols, evidence labels, missing-experiment
specifications, and all 21 source tables are retained in
`tables_experiment_catalog.md`.

| Final table | Report | Slides | Slide treatment |
|---|:---:|:---:|---|
| R1. Dataset and evaluation setting | Yes | No | Show dataset scale as large-number callouts, not a table |
| R2. Legal-structure-aware chunking | Yes | Yes — S1 | Keep Article, Token window, Khoản, and Khoản–Điểm rows; retain Macro-F2 and Recall@100 |
| R3. First-stage retrieval selection | Yes | Yes — S2 | Show the six controlled component combinations; retain Macro-F2 and Recall@50 |
| R4. Structure and cross-reference ablation | Yes | Yes — S3 | Keep the four factorial rows and Macro-F2; if incomplete, show candidate-recall gate with limitation |
| R5. Reranker adaptation and candidate budget | Yes | No | Supporting checkpoint-selection detail; feed its winner into S4 |
| R6. LightGBM feature design | Yes | No | Supporting learning-to-rank detail; feed its winner into S4 |
| R7. Final-set decision policy | Yes | Yes — S4 | Keep fixed top 2, raw threshold, score gap, and oracle; combine with selected R5/R6 milestones |
| R8. Official benchmark and hidden-split robustness | Yes | Yes — S5 | Keep organizer baseline, at most two published systems, and three project policies |
| R9. Error attribution and computational cost | Yes | No | Show as a failure funnel or examples rather than a dense slide table |

Blank cells are intentional and mean that the controlled experiment or
measurement has not yet been completed. Development values use the persisted
seed-42 split of 438 queries unless stated otherwise. Public/private labels are
never used for tuning.

---

## R1. Dataset and Evaluation Setting

**Use:** `REPORT ONLY`

**Question.** What is the query scale, question-length distribution, and
variable output-set requirement across the official DRiLL splits?

**Protocol.** Values in the three official split columns follow Table 2 of the
DRiLL shared-task overview. Word counts use the organizer's question-level
counting convention and are descriptive rather than tokenizer lengths. Public
and private relevance labels remain hidden and are never used for tuning. The
project development set is a persisted seed-42 subset of 438 training queries.

| Statistic | Train | Public test | Private test |
|---|---:|---:|---:|
| Number of questions | 2,190 | 312 | 627 |
| Words per question — average | 19.71 | 20.00 | 19.90 |
| Words per question — minimum | 6 | 11 | 11 |
| Words per question — maximum | 45 | 42 | 44 |
| Relevant articles/question — average | 1.34 | 1.31 | 1.32 |
| Relevant articles/question — minimum | 1 | 1 | 1 |
| Relevant articles/question — maximum | 9 | 4 | 5 |

The accompanying corpus contains 2,157 legal documents and 59,636 raw
article-level retrieval units; 59,628 non-blank articles are indexed. Article
IDs (`aid`) are the prediction unit. These corpus statistics are stated in
prose to avoid mixing corpus-only fields with query-split columns.

**Source:** `docs/dataset.md`, `docs/evaluation.md`, official DRiLL overview.

---

## R2. Effect of Legal-Structure-Aware Chunking

**Use:** `REPORT + SLIDES` (`S1`)

**Question.** Do legal Khoản/Điểm boundaries improve dense retrieval relative
to article truncation and generic token windows?

**Protocol.** Rerun all four rows from scratch with the primary dense component
`mainguyen9/vietlegal-harrier-0.6b`. Freeze its model revision, model-card query
instruction, document prefix, pooling, maximum sequence length, exact cosine
search, max chunk-to-article aggregation, candidate depth 100, persisted
seed-42 development qids, and evaluator. Use fixed top 2 for official
Precision/Recall/Macro-F2 so this table is aligned with downstream checkpoint
selection. Report Recall@10/50/100 from the same saved depth-100 rankings.

The only treatment variable is representation: article truncation, fixed token
window, Khoản, or hierarchical Khoản–Điểm. Article truncation emits one full
article record and is truncated by the frozen 512-token encoder limit. The
three chunked treatments use a 448-token document budget and 64-token
overlap/fallback measured with the frozen BGE-M3 boundary tokenizer, then use
VietLegal-Harrier for embedding, consistently with the selected pipeline.
Fixed token windows are the sole non-structural chunking baseline; recursive
windows remain historical appendix evidence and are not rerun.

| Representation | Precision@2 | Recall@2 | Macro-F2@2 | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| Article truncation | 0.3368 | 0.5732 | 0.5026 | 0.7964 | 0.9125 | 0.9422 |
| Fixed token window 448/64 | **0.3425** | **0.5808** | **0.5098** | **0.8166** | **0.9256** | **0.9517** |
| Khoản + fallback 448/64 | 0.3345 | 0.5742 | 0.5022 | 0.7993 | 0.9186 | 0.9471 |
| Hierarchical Khoản–Điểm 448/64 | 0.3379 | 0.5811 | 0.5079 | 0.8031 | 0.9244 | 0.9463 |

The fixed-window control is the point-estimate winner. Hierarchical
Khoản–Điểm improves over article truncation by `+0.0053` Macro-F2@2 and
`+0.0119` Recall@50, but trails fixed windows by `-0.0019` Macro-F2@2 and
`-0.0012` Recall@50. These small differences require paired query-level
uncertainty analysis before making a significance claim.

**Slide S1:** after all four rows complete, show Representation, Macro-F2@2,
and Recall@100. Every slide value must be copied unchanged from R2.

**Source:** complete user-executed Table-R2 notebook runs reported on
2026-11-01; detailed Table 4 in `tables_experiment_catalog.md`. Provenance and
paired verification remain pending ingestion of the result archive.

---

## R3. Selection of the First-Stage Retrieval Pipeline

**Use:** `REPORT + SLIDES` (`S2`)

**Question.** Which sparse/dense components provide the strongest and most
complementary first-stage candidate ranking without turning the study into a
model zoo?

**Protocol.** Freeze the seed-42 split, BGE-M3-tokenized hierarchical
Khoản–Điểm 448/64 chunk store, model-card prefixes, exact cosine search, max
chunk-to-article aggregation, component depth 100, and evaluator. Dense 1 is
VietLegal-Harrier and Dense 2 is multilingual-E5-large. Combination rows reuse
the exact saved component rankings and WRRF `k=10`; their relative weights are
inherited from the frozen final ratio BM25/Harrier/E5 `0.15/0.55/0.30` and
renormalized only over the enabled components. No reranker, reference
expansion, or LightGBM is allowed in this table. Rerun all six rows from the
same saved component rankings rather than mixing historical runs. Report the
same fixed top-2 Precision/Recall/Macro-F2 and Recall@50 for every row because
50 articles is the frozen downstream reranker budget; retain Recall@100 only
in the detailed artifact as a depth diagnostic.

The exact enabled-component weights are: BM25 `1.0`; Harrier `1.0`; E5 `1.0`;
BM25+Harrier `0.21/0.79`; Harrier+E5 `0.65/0.35`; and the full hybrid
`0.15/0.55/0.30`. The two-component weights are rounded, declared controls
derived from the final ratio rather than separately tuned optima.

| First-stage components | WRRF weights over enabled components | Precision@2 | Recall@2 | Macro-F2@2 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| BM25 + VnCoreNLP | `1.0` | 0.2511 | 0.4297 | 0.3762 | 0.8348 |
| Dense 1: VietLegal-Harrier | `1.0` | 0.3379 | 0.5811 | 0.5079 | 0.9244 |
| Dense 2: multilingual-E5-large | `1.0` | 0.3116 | 0.5293 | 0.4644 | 0.8960 |
| BM25 + Dense 1 | `0.21 / 0.79` | 0.3402 | 0.5795 | 0.5080 | 0.9243 |
| Dense 1 + Dense 2 | `0.65 / 0.35` | 0.3505 | 0.5974 | 0.5236 | 0.9244 |
| **BM25 + Dense 1 + Dense 2** | `0.15 / 0.55 / 0.30` | **0.3516** | **0.5977** | **0.5243** | **0.9278** |

**Slide S2:** use these six rows with Method, Macro-F2@2, and Recall@50. The
row order communicates the component-building logic, not a guaranteed
monotonic progression.

**Source:** complete user-executed Table-R3 notebook cases reported on
2026-11-01. All rows were derived offline from the same saved component
rankings and development qids. The result archive remains to be ingested for
independent provenance verification. These are matched component ablations,
but the small observed differences do not establish statistical significance.

---

## R4. Structure and Cross-Reference Ablation

**Use:** `REPORT + SLIDES` (`S3`)

**Question.** Do structure-aware chunks and deterministic same-document legal
references independently improve the strong retrieval/reranking pipeline?

**Protocol.** Freeze split, encoders, WRRF, candidate budget, reranker,
LightGBM schema, final policy, and evaluator. The non-structural control is the
fixed token window 448/64 selected by the point estimates in R2; the structural
treatment is hierarchical Khoản–Điểm 448/64. References use graph v3,
same-document scope, top-five seeds, maximum four additions, and fixed-budget
displacement. Thus each row changes only chunk representation, reference
expansion, or both. Candidate recall and nested final metrics must both be
reported.

| Chunk representation | Same-document references | Candidate Recall@50 | Reranker F2@2 | Precision | Recall | Nested-OOF Macro-F2 |
|---|---|---:|---:|---:|---:|---:|
| Fixed token window 448/64 | Off | 0.9341 | **0.5937** | 0.5179 | **0.7204** | **0.6681** |
| Hierarchical Khoản–Điểm 448/64 | Off | 0.9042 | 0.5799 | **0.5327** | 0.7014 | 0.6597 |
| Fixed token window 448/64 | On | **0.9440** | **0.5937** | 0.5148 | 0.7200 | 0.6669 |
| Hierarchical Khoản–Điểm 448/64 | On | 0.9178 | 0.5799 | 0.5323 | 0.6997 | 0.6583 |

The selected historical pipeline is reproduced by the hierarchical/reference-
on cell (`0.6583`, rounding the prior `0.658333`). Fixed windows outperform
hierarchical chunks by `0.0084` without references and `0.0086` with references
on nested Macro-F2. References raise candidate Recall@50 by `0.0099` for fixed
windows and `0.0136` for hierarchical chunks, but reduce nested Macro-F2 by
`0.0012` and `0.0014`. The difference-in-differences is only about `0.0002`, so
the point estimates provide no meaningful structure/reference interaction.
Paired uncertainty remains required before claiming significance.

**Slide S3:** show the four completed cells and highlight that graph expansion
improves candidate recall but not final F2, while fixed windows are consistently
stronger under both graph settings.

**Source:** complete user-executed Table-R4 notebook rows reported on 2026-11-01;
independent archive/hash ingestion remains pending. Detailed Tables 8 and 9 in
`tables_experiment_catalog.md` retain supporting graph-gate provenance.

---

## R5. Reranker Adaptation and Candidate-Budget Comparison

**Use:** `REPORT ONLY`

**Question.** Which adaptation checkpoint is best when threshold scale is
removed from model selection, and is pool 100 better than pool 50?

**Protocol.** Reuse one checksummed frozen handoff per pool. Load warm start and
adaptation checkpoints in fresh subprocesses, score identical candidates, max
aggregate chunk logits to `aid`, and select by fixed top 2. Threshold results
are diagnostics, not checkpoint-selection metrics. Training/mining/graph/WRRF
settings are fixed across pools except candidate budget.

| Pool | Snapshot | Fixed top-2 Precision | Fixed top-2 Recall | Fixed top-2 Macro-F2 | Pooled threshold Macro-F2 | Decision |
|---:|---|---:|---:|---:|---:|---|
| 50 | Vietnamese warm start | 0.367580 | 0.615830 | 0.542546 | 0.585978 | Baseline |
| 50 | Epoch 1 / `1208` | **0.393836** | **0.657610** | **0.579928** | **0.623556** | Selected finalist |
| 50 | Epoch 5 / `6054` | 0.391553 | 0.654566 | 0.577044 | 0.621791 | Late-epoch control |
| 100 | Epoch 2 / `2479` | 0.385845 | 0.639612 | 0.565259 | 0.601615 | Pool-100 fixed-top-2 winner |

**Source:** artifact-verified checkpoint archives; the complete 12-row sweep is
Table 11 in `tables_experiment_catalog.md`.

---

## R6. Learning-to-Rank Feature Design

**Use:** `REPORT ONLY`

**Question.** Which feature schema improves the frozen neural ranking, and do
exploratory group removals justify a smaller confirmatory schema?

**Protocol.** Freeze pool-50 checkpoint `1208`, 13,348 candidate rows with 522
positives, the same 438 qids, five query-grouped folds, seed 42, LightGBM
hyperparameters, and score-gap search. Schemas are nested 15/41/61/69-feature
supersets. Group removal starts from the 61-feature schema and changes one
predefined group only. Multiple removals are exploratory, not confirmatory.

| Treatment | Features | Fixed top-2 Macro-F2 | Pooled threshold Macro-F2 | Nested score-gap Macro-F2 | Interpretation |
|---|---:|---:|---:|---:|---|
| `base_v1` | 15 | 0.583262 | 0.664100 | 0.648388 | Historical base features |
| `enhanced_v1` | 41 | 0.594233 | **0.669570** | 0.649468 | Retrieval/query features |
| `enhanced_v2` | 61 | 0.594233 | 0.663271 | **0.658333** | Selected full schema |
| `enhanced_v3` | 69 | **0.597850** | 0.662563 | 0.657139 | Static article features added |
| `enhanced_v2` minus retriever agreement | Reduced |  |  | 0.661736 | Exploratory; one confirmation permitted |
| `enhanced_v2` minus query/chunk lexical | Reduced |  |  | 0.660375 | Exploratory only |
| `enhanced_v2` minus within-query calibration | Reduced |  |  | 0.640962 | Strong negative removal control |

**Source:** artifact-verified feature-screen archive; detailed Tables 12 and 13
in `tables_experiment_catalog.md`.

---

## R7. Final-Set Decision Policy Comparison

**Use:** `REPORT + SLIDES` (`S4`)

**Question.** Given one frozen LightGBM ranking, how should the system choose
the number of returned articles per query?

**Protocol.** Use the same `enhanced_v2` OOF rankings for all policies. Fit or
tune every learned policy without its outer validation fold. Fixed top-k needs
no tuning. Oracle prefix is non-deployable. Report output-size behavior before
adopting a new selector.

| Policy | Precision | Recall | Macro-F2 | Mean output size | Maximum output size | Role |
|---|---:|---:|---:|---:|---:|---|
| Fixed top 1 | 0.652968 | 0.565145 | 0.580767 | 1.000 | 1 | Precision-oriented control |
| Fixed top 2 | 0.402968 | 0.674239 | 0.594233 | 2.000 | 2 | Primary fixed control |
| Nested raw threshold + fallback 2 | 0.499125 | **0.713052** | 0.656754 | 2.235 | 6 | Threshold candidate |
| Nested score gap | **0.532344** | 0.699734 | **0.658333** |  | 3 | Current scientific estimate |
| Logistic gold-cardinality predictor | 0.338128 | 0.732002 | 0.593688 | 3.397 | 5 | Rejected baseline |
| Learned stopping policy |  |  |  |  | 5 | Nested-OOF values must be copied from the saved artifact; official results are in R8 |
| Multiclass logistic cardinality | 0.556355 | **0.705441** | **0.669557** | 2.315 | 5 | Closed-screen nested-OOF winner; C=0.1, balanced |
| Tuned prefix-utility Ridge | **0.574658** | 0.697184 | 0.668670 | 2.039 | 5 | Close second; alpha=1.0 |
| Oracle prefix through k=5 | 0.722793 | 0.794809 | 0.779280 | 1.610 | 5 | Upper bound only |

Platt and isotonic calibrated thresholds both measured the same F2 `0.656754`
as the raw nested threshold and are omitted from this compact table because a
monotonic score transform adds no information to an exact threshold search.
The multiclass winner exceeds tuned Ridge by only `0.000887` F2. Its exact-k,
under-selection, and over-selection rates are `0.4840/0.1301/0.3858`; higher
over-selection is consistent with the recall-weighted F2 objective.

**Slide S4:** show four milestones: reranker fixed top 2 (`0.579928`, from R5),
LightGBM fixed top 2 (`0.594233`, from R6), nested threshold (`0.656754`), and
nested score gap (`0.658333`). Clearly label this as pipeline storytelling, not
a single-variable ablation.

**Source:** verified policy screen and detailed Table 14 in
`tables_experiment_catalog.md`.

---

## R8. Official Benchmark and Hidden-Split Robustness

**Use:** `REPORT + SLIDES` (`S5`)

**Question.** How does the frozen project pipeline compare with published DRiLL
systems, and are its policy results stable across public/private splits?

**Protocol.** Published private results are external context, not controlled
baselines. Project rows freeze retrieval, checkpoint `1208`, `enhanced_v2`, and
policy parameters before submission. Public/private results cannot select a new
method. Missing project Precision/Recall remain blank because Codabench output
provided only F2 here.

| System / policy | Evidence type | Public Macro-F2 | Private Precision | Private Recall | Private Macro-F2 | Submission / source |
|---|---|---:|---:|---:|---:|---|
| Organizer BM25 | Published external |  | 0.2265 | 0.3830 | 0.3365 | Official overview |
| Organizer two-stage, fixed top 2 | Published external |  | 0.3740 | 0.6253 | 0.5512 | Official overview |
| ViDRILL | Published external |  | 0.4153 | 0.7605 | 0.6521 | Team paper / official table |
| CUDO | Published external |  | 0.5097 | 0.7653 | 0.6955 | Team paper / official table |
| FPT IS | Published external |  | 0.6222 | 0.7181 | 0.6966 | Team paper / official table |
| EDM | Published external |  | 0.6773 | 0.7394 | 0.7261 | Team paper / official table |
| This project, fixed top 2 | Codabench | 0.5989 |  |  | 0.5552 | Public 1022898; private 1022897 |
| This project, threshold + fallback 2 | Codabench | 0.6524 |  |  | **0.6054** | Public 1021667; private 1021640 |
| This project, score gap | Codabench | **0.6551** |  |  | 0.6047 | Public 1021732; private 1021639 |
| This project, fold-safe learned stopping | Codabench | **0.6672** |  |  | **0.6123** | Public 1025830; private 1025816 |
| This project, listwise prefix-utility Ridge | Codabench | 0.6661 |  |  | **0.6245** | Public 1025919; private 1025907 |
| This project, tuned prefix-utility Ridge | Codabench | 0.6700 |  |  | 0.6265 | Public 1025993; private 1025992 |
| This project, balanced multiclass logistic | Codabench | **0.6847** |  |  | **0.6368** | Public 1025984; private 1025979 |

**Slide S5:** keep organizer two-stage, at most two published systems (normally
ViDRILL and EDM), and the four project-policy rows. Do not display a project
rank because protocols and competition timing differ.

**Source:** `docs/competition_review.md`, `docs/experiments.md`, detailed Tables
16 and 17 in `tables_experiment_catalog.md`.

---

## R9. Error Attribution and Computational Cost

**Use:** `REPORT ONLY`

**Question.** Where do remaining errors arise, and what computational cost is
paid by the selected pipeline?

**Protocol.** Use the same frozen qids and saved artifacts at every stage.
Candidate miss, reranker demotion, LightGBM changes, and under/over-selection
must have deterministic definitions. Efficiency runs require named hardware,
cold/cache-hit status, wall time, throughput, process-tree RAM, per-device peak
VRAM, and artifact size. Do not compare different environments as if they were
matched.

| Audit type | Stage / category | Definition or workload | Primary count / cost | Secondary diagnostic | Environment / evidence |
|---|---|---|---:|---|---|
| Error | Candidate miss | At least one gold aid absent from first-stage top 50 |  | Mean missed gold:  | Saved first-stage rankings |
| Error | Reranker demotion | Gold present but moved below deployable region |  | Macro-F2 loss:  | Chunk candidates + reranker scores |
| Error | LightGBM degradation | Gold rank worsens from reranker to LightGBM |  | Macro-F2 loss:  | Reranker + OOF rankings |
| Error | LightGBM recovery | Gold rank improves from reranker to LightGBM |  | Macro-F2 contribution:  | Reranker + OOF rankings |
| Error | Under-selection | Ranked gold excluded by final policy |  | Mean missed gold:  | Final-policy predictions |
| Error | Over-selection | Final set includes avoidable false positives |  | Mean false positives:  | Final-policy predictions |
| Cost | Native BM25, uncached | top 100 × 438 | 341.36 s | Artifact: 73.78 MiB | Local CPU, Python 3.12.9 |
| Cost | bm25s, uncached | top 100 × 438 | 295.43 s | Artifact: 55.26 MiB | Local CPU, Python 3.12.9 |
| Cost | Native BM25, cache hit | top 100 × 438 | 37.49 s | Artifact: 73.78 MiB | Local CPU, Python 3.12.9 |
| Cost | bm25s, cache hit | top 100 × 438 | 9.16 s | Artifact: 55.26 MiB | Local CPU, Python 3.12.9 |
| Cost | Selected multi-dense retrieval | Frozen 438-query dev run |  | Throughput/RAM/VRAM:  | Kaggle T4×2 + FAISS CPU |
| Cost | Pointwise reranker, pool 50 | Frozen candidates, batch 256 |  | Pairs/s and VRAM:  | Kaggle T4×2 |
| Cost | LightGBM, 61 features | 13,348 rows |  | Training time/RAM:  | CPU |

**Source:** detailed Tables 18–20 in `tables_experiment_catalog.md`. R9 remains
incomplete until the consolidated error-analysis and resource audit run.

---

## Slide Extraction Checklist

Slides must reuse report values, not introduce new calculations:

1. **S1 from R2:** four chunking rows; Macro-F2 and Recall@100 only.
2. **S2 from R3:** six controlled first-stage component rows; Macro-F2 and
   Recall@50 only; do not imply monotonic gains.
3. **S3 from R4:** four factorial rows when complete; otherwise the explicitly
   labelled reference recall gate.
4. **S4 from R5–R7:** four pipeline milestones; label non-causal progression.
5. **S5 from R8:** organizer two-stage, up to two published systems, and three
   project policies.

Any later correction to a result must be made in the report table first, then
propagated verbatim to its slide view.

## Recommended Next Execution Order

1. Run the complete four-row R2 table with Harrier, fixed top 2, and identical
   candidate/evaluation settings. Build deterministic representation-specific
   chunk caches and retain rankings, metrics, configs, split qids, and runtime.
2. Consolidate missing historical artifacts supporting R3 and the appendix;
   historical E5-base/top-3 chunking values are context only and do not populate
   the new R2.
3. Run the R9 error audit before introducing another model.
4. Confirm exactly one reduced LightGBM schema for R6.
5. Evaluate one information-adding learned stopping policy for R7.
6. Complete the two missing structure/reference factorial cells in R4 if the
   primary research claim is retained.
7. Run one bounded, predeclared LightGBM sensitivity grid only after R6/R7 are
   frozen.
8. Collect clean-stage runtime/RAM/VRAM measurements for R9.
9. Submit only a development-selected frozen method; update R8 afterward.
