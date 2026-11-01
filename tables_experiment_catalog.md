# Experimental Evidence Catalogue for Vietnamese Legal Retrieval

The nine report-ready tables and the four slide extractions are maintained in
`tables_experiment.md`. This catalogue retains the 21 detailed source tables,
protocols, evidence levels, and execution specifications used to construct
those final tables.

## Scope, evidence levels, and reporting rules

This document is the execution blueprint for the final report and presentation.
The research question remains whether legal structure and explicit
cross-provision references improve Vietnamese legal retrieval over a strong
hybrid-plus-reranking baseline. It is not a catalogue of every model that could
be run.

All project development values use the persisted seed-42 split of 438 queries
unless a table says otherwise. Official Precision and Recall are macro-averaged
over queries, and Macro-F2 is computed from those two macro values. Public and
private test results are reporting evidence only; they must not select a model,
feature, threshold, or output policy.

Evidence labels used below:

- **Artifact-verified:** a retained repository JSON/CSV/tar artifact supports
  the value.
- **Recorded:** a source-of-truth project document records the executed value,
  but the original run bundle is no longer in the checkout.
- **User-reported / pending verification:** the run was reported by the owner,
  but its resolved config, split IDs, and predictions still need inspection.
- **Published external:** copied from a cited paper; not measured by this
  repository.

Blank cells mean that the experiment or measurement has not been completed.
They are intentionally not filled with estimates.

External design references are the official DRiLL overview and the published
EDM, FPT IS, CUDO, and ViDRILL system papers listed in
`docs/competition_review.md`. They motivate stage-wise retrieval, reranking,
learning-to-rank, hierarchical chunking, and adaptive output selection, but the
tables below are tailored to this repository's pipeline.

Primary external references:

- Vuong et al., official DRiLL overview:
  <https://aclanthology.org/2025.vlsp-1.16/>.
- Tran et al., ViDRILL multi-stage retrieval:
  <https://aclanthology.org/2025.vlsp-1.17/>.
- Hoai et al., FPT IS hierarchical chunking:
  <https://aclanthology.org/2025.vlsp-1.18/>.
- Pham et al., EDM learning-to-rank pipeline:
  <https://aclanthology.org/2025.vlsp-1.19/>.
- Hieu and Do, CUDO/FPT.AI two-stage framework:
  <https://aclanthology.org/2025.vlsp-1.20/>.

## How to use this document

The 21 numbered tables below are an **experiment evidence catalogue**, not a
requirement to place 21 tables in a 15-page report. They preserve protocols,
screening evidence, missing cells, and provenance so report tables can be built
without reopening methodology decisions.

The final deliverables use three layers:

1. **Main report:** nine compact, consolidated tables (`R1`–`R9`).
2. **Presentation:** four simplified views (`S1`–`S4`), normally one per slide.
3. **Evidence catalogue / appendix:** Tables 1–21 in this file. These are the
   detailed sources from which the compact report and slide tables are derived.

### Main-report layout for a 15-page report

| Report table | Final title | Evidence catalogue source | Target size | Role in the argument |
|---|---|---|---|---|
| R1 | Dataset and Evaluation Setting | Tables 1 and 21 | 4 rows, about 1/3 page | Define task, splits, variable evidence-set size, and reproducibility boundary |
| R2 | Effect of Legal-Structure-Aware Chunking | Table 4 | 5 rows, about 1/2 page | Establish the structure-aware representation result |
| R3 | Selection of the First-Stage Retrieval Pipeline | Tables 2, 5, and 6 | 4–5 rows, about 1/2 page | Compare the sparse anchor, dense anchor, legal dense winner, and selected multi-dense fusion |
| R4 | Structure and Cross-Reference Ablation | Tables 8 and 9 | Four factorial rows, about 1/2 page | Answer the primary research question; incomplete cells must remain explicit until run |
| R5 | Reranker Adaptation and Candidate-Budget Comparison | Tables 7 and 11 | Warm start plus three finalists, about 1/2 page | Select checkpoint and pool without threshold leakage |
| R6 | Learning-to-Rank Feature and Ablation Summary | Tables 12 and 13 | Four schemas plus at most three removals, about 2/3 page | Show the value and limits of LightGBM features |
| R7 | Final-Set Decision Policy Comparison | Table 14 | Fixed controls plus deployable policies and oracle, about 1/2 page | Explain the largest remaining quality gain |
| R8 | Official Benchmark and Hidden-Split Robustness | Tables 16 and 17 | 6–8 rows, about 2/3 page | Separate published external context from project public/private results |
| R9 | Error Attribution and Computational Cost | Tables 18–20 | One compact error block plus one compact cost block, about 3/4 page | Explain remaining failures and practical trade-offs |

Tables R1–R9 should occupy roughly four to five report pages in total. Sparse
parameter sweeps, full model screening, all checkpoints, all feature removals,
and detailed provenance belong in an appendix or artifact bundle.

### Presentation layout

| Slide view | Content | Source | Presentation rule |
|---|---|---|---|
| S1 | First-stage and legal-structure progression | R2–R3 | Show at most four methods; use one highlighted takeaway |
| S2 | Main structure/reference ablation | R4 | Use the four factorial cells; if incomplete, show the reference recall gate instead and state the limitation |
| S3 | Reranker → LightGBM → adaptive selection progression | R5–R7 and Table 15 | Use one compact progressive table or chart; label it storytelling rather than causal ablation |
| S4 | Final public/private result and benchmark context | R8 | Keep organizer baseline, one or two published systems, fixed top 2, and the two frozen adaptive policies |

Dataset scale should normally be displayed as three or four large numbers rather
than a separate table slide. Error analysis is better shown as a small failure
funnel or two audited examples; the full R9 table remains in the report.

---

# Evidence Catalogue and Appendix Specifications

## A. Dataset, Sparse Retrieval, and Representation Screening

---

## Table 1. Dataset, Split, and Evidence-Set Statistics

**Status:** `REPORT ONLY`

**Purpose.** Establish the task scale, distinguish documents from article-level
retrieval units, and show why fixed top-k is structurally mismatched to a
variable-size evidence-set task.

**Protocol.** Corpus values come from the verified 2026-09-28 local snapshot.
Train/public/private query counts and test evidence-set statistics come from the
official DRiLL report. Do not infer labels for public/private. Report both raw
articles and indexable non-blank articles.

| Collection / split | Legal documents | Raw articles | Indexed non-blank articles | Queries | Mean relevant articles | Maximum relevant articles | Label availability |
|---|---:|---:|---:|---:|---:|---:|---|
| Legal corpus | 2,157 | 59,636 | 59,628 |  |  |  | Corpus only |
| Train |  |  |  | 2,190 | 1.34 | 9 | Available |
| Persisted development subset |  |  |  | 438 |  |  | Available |
| Public test |  |  |  | 312 | 1.31 | 4 | Hidden |
| Private test |  |  |  | 627 | 1.32 | 5 | Hidden |

**Evidence.** `docs/dataset.md`; official values are published external.

---

## Table 2. Effect of Vietnamese Tokenization on BM25

**Status:** `REPORT ONLY`

**Purpose.** Determine whether Vietnamese word segmentation improves the sparse
anchor before adding dense retrieval.

**Protocol.** Hold the 438-query split, corpus, Unicode NFKC normalization,
lowercasing, BM25 `k1=1.2`, `b=0.75`, candidate depth 100, final top 3, and
official evaluator fixed. Only the tokenizer changes. Dependency versions and
VnCoreNLP JAR checksum must be recorded in any confirmation run.

| Tokenization | Precision | Recall | Macro-F2 | Recall@10 | Recall@50 | Recall@100 | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| Regex/NFKC | 0.180365 | 0.452593 | 0.347650 | 0.637872 | 0.804066 | 0.844156 | Recorded |
| PyVi | 0.196347 | 0.490645 | 0.377485 | 0.669129 | **0.842803** | **0.878593** | User-reported / pending verification |
| Underthesea | 0.194825 | 0.491977 | 0.376981 | 0.673695 | 0.835192 | 0.876310 | User-reported / pending verification |
| VnCoreNLP | **0.196347** | **0.495782** | **0.379908** | **0.674837** | 0.834758 | 0.877452 | Recorded |

**Evidence.** `docs/experiments.md`, BM25 tokenization ablation.

---

## Table 3. Sparse Retrieval Configuration and Sensitivity

**Status:** `SUPPLEMENTARY / OPTIONAL`

**Purpose.** Verify that the sparse anchor is not an arbitrary BM25 setting and
determine whether BM25+ helps under the same tokenizer and evaluation budget.

**Protocol.** Use VnCoreNLP, the same split and article index, candidate depth
100, final top 3, identical evaluator, and no dense component. For parameter
sensitivity, vary only `k1` and `b`. BM25+ changes only the scoring variant with
`delta=1.0`.

| Sparse method | k1 | b | Output k | Precision | Recall | Macro-F2 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 1.2 | 0.75 | 3 | 0.196347 | 0.495782 | 0.379908 | 0.834758 | **0.877452** |
| BM25, alternative setting | 1.5 | 0.75 | 3 |  |  | **0.381754** | **0.838182** | 0.877071 |
| BM25+ | 1.2 | 0.75 | 3 | 0.192542 | 0.484176 | 0.371605 | 0.814264 | 0.853783 |

**Evidence.** `docs/experiments.md`. The complete 12-setting sensitivity grid
should remain supplementary rather than occupy the main report.

---

## Table 4. Chunking and Retrieval Granularity under a Fixed Dense Encoder

**Status:** `REPORT + SLIDES`

**Purpose.** Test whether legal boundaries improve retrieval over article
truncation and generic token windows while holding the encoder fixed.

**Protocol.** Use `mainguyen9/vietlegal-harrier-0.6b`, its model-card query
instruction, BGE-M3 chunk boundaries, exact cosine search, max
chunk-to-article aggregation, candidate depth 100, fixed top 2, seed-42 split,
and the common evaluator. Article truncation uses the frozen 512-token encoder
limit; the three chunked treatments use 448/64. Only representation boundaries
change. Values are complete user-executed measurements reported on 2026-11-01
and remain pending independent result-archive verification.

| Representation | Precision@2 | Recall@2 | Macro-F2@2 | Recall@10 | Recall@50 | Recall@100 | Runtime (s) | Evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Article truncation | 0.3368 | 0.5732 | 0.5026 | 0.7964 | 0.9125 | 0.9422 | 3332.47 | User-reported / pending archive verification |
| Fixed token window 448/64 | **0.3425** | 0.5808 | **0.5098** | **0.8166** | **0.9256** | **0.9517** | 5798.79 | User-reported / pending archive verification |
| Khoản + fallback 448/64 | 0.3345 | 0.5742 | 0.5022 | 0.7993 | 0.9186 | 0.9471 | 5812.49 | User-reported / pending archive verification |
| Hierarchical Khoản–Điểm 448/64 | 0.3379 | **0.5811** | 0.5079 | 0.8031 | 0.9244 | 0.9463 | 5431.82 | User-reported / pending archive verification |

**Evidence.** `docs/experiments.md` and the four Table-R2 notebook outputs.
The earlier multilingual-E5-base/top-3 chunking screen is historical evidence,
not part of this controlled report table.

---

## Table 5. Single Dense Retriever Screening

**Status:** `REPORT ONLY`

**Purpose.** Select a small set of complementary dense encoders rather than
construct a model zoo. The anchor is BGE-M3, multilingual control is E5-large,
and Vietnamese legal-specific candidates test the value of domain adaptation.

**Protocol.** Freeze BGE-M3-tokenized Khoản–Điểm 448/64 chunks, exact cosine
search, model-card query/document prefixes, max article aggregation, candidate
depth 100, seed-42 split, and evaluator. Each encoder receives its documented
maximum sequence length and pooling. Compare final output at each model's
recorded best fixed k, while candidate recall remains independent of that k.

| Role | Dense encoder | Best output k | Precision | Recall | Macro-F2 | Recall@50 | Recall@100 | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Multilingual anchor | `BAAI/bge-m3` | 2 | 0.309361 | 0.526826 | 0.461889 | 0.878816 | 0.910421 | Recorded, original bundle removed |
| Multilingual control | `intfloat/multilingual-e5-large` | 2 | 0.311644 | 0.529322 | 0.464441 | 0.896015 | 0.929447 | Recorded, original bundle removed |
| Vietnamese general | `nhminh107/VietRAG-Embed` | 2 | 0.286530 | 0.489346 | 0.428661 | 0.848543 | 0.883496 | Recorded, original bundle removed |
| Vietnamese legal E5-large | `mainguyen9/vietlegal-e5` | 3 | 0.242770 | 0.610997 | 0.468788 | 0.919145 | **0.946760** | Recorded, original bundle removed |
| Vietnamese legal instruction | `mainguyen9/vietlegal-harrier-0.6b` | 2 | **0.337900** | 0.581050 | **0.507947** | **0.924364** | 0.946325 | Recorded, original bundle removed |
| Vietnamese legal E5-base | `bqbbao6/vietnamese-legal-embedding` | 2 | 0.311644 | 0.519597 | 0.458418 | 0.897380 | 0.928800 | Recorded, original bundle removed |
| Vietnamese legal XLM-R/HaLong | `truro7/vn-law-embedding` | 2 | 0.287671 | 0.483828 | 0.425764 | 0.859252 | 0.909426 | Recorded, original bundle removed |

**Evidence.** `docs/experiments.md`. Future combinations should retain only
Harrier plus one complementary multilingual or high-recall peer.

---

## Table 6. Dense-Pair and Weighted-RRF Selection

**Status:** `REPORT + SLIDES`

**Purpose.** Determine which second dense encoder best complements Harrier
after sparse+dense fusion and downstream ranking.

**Protocol.** Freeze BM25S/VnCoreNLP, shared Khoản–Điểm 448/64 chunks, the same
40-case WRRF grid, component depth 100, candidate budget 50, epoch-3 reranker,
fresh five-fold query-grouped LightGBM, nested score-gap evaluation, and split.
The winning WRRF setting may differ per pair, so this is an end-to-end pair
screen, not an encoder-only ablation.

| Dense pair | Fused Recall@20 | Fused Recall@50 | Fused Recall@100 | Nested-OOF Macro-F2 | Selection outcome |
|---|---:|---:|---:|---:|---|
| Harrier + multilingual-E5-large | 0.875826 | 0.927789 | **0.952794** | **0.648354** | Selected |
| Harrier + VietLegal-E5 |  | **0.933823** | 0.952033 | 0.630950 | Higher R@50, weaker early/downstream ranking |
| Harrier + BGE-M3 | 0.878164 | 0.926593 | 0.949369 | 0.629978 | Not selected |

**Evidence.** `docs/research_plan.md`, measured three-pair dense selection.

---

## Table 7. Candidate Depth and Reranking Budget

**Status:** `REPORT ONLY`

**Purpose.** Quantify the recall gained by deeper candidates against the near
linear increase in cross-encoder work.

**Protocol.** Freeze the selected BM25/Harrier/E5-large WRRF ranking,
preprocessing, split, and candidate ordering. Candidate Recall@k is computed
before reranking. For the controlled pool-50/pool-100 comparison, hold training,
mining, reranker architecture, and fixed-top-2 checkpoint gate constant.

| Candidate budget | First-stage Recall | Approximate reranker pairs/query | Best checkpoint by fixed top 2 | Best fixed-top-2 Macro-F2 | Best pooled threshold Macro-F2 |
|---:|---:|---:|---|---:|---:|
| 20 | 0.875826 | 20 |  |  |  |
| 50 | 0.927789 | 50 | Pool-50 epoch 1 / `1208` | **0.579928** | **0.623556** |
| 100 | 0.952794 | 100 | Pool-100 epoch 2 / `2479` | 0.565259 | 0.610202 |

**Evidence.** `docs/research_plan.md`; checkpoint values are artifact-verified
in the two checkpoint-evaluation archives. Candidate budget is not identical to
the number of packed chunks for every query, so actual pair counts should be
added from provenance when reporting efficiency.

---

## B. Main Structure, Reference, Reranking, and Selection Experiments

## Table 8. Explicit Cross-Reference Expansion Recall Gate

**Status:** `REPORT + SLIDES`

**Purpose.** Test whether deterministic legal references recover missing gold
articles before spending reranker compute, and whether same-document or
cross-document edges supply the useful signal.

**Protocol.** Freeze the retuned selected first-stage ranking, cutoff 50, seed
depth 5, maximum four expansions, graph v3, and fixed-budget displacement.
Change only edge scope. This is a retrieval-stage diagnostic and must not be
presented as end-to-end F2.

| Expansion scope | Expanded queries | Mean additions | Recovered gold pairs | Dropped gold pairs | Recall@50 fixed budget | Change vs no expansion |
|---|---:|---:|---:|---:|---:|---:|
| None | 0 | 0 | 0 | 0 | 0.927789 | 0.000000 |
| Same document | 172 | 0.821918 | 21 | 1 | 0.948174 | +0.020385 |
| Cross document | 24 | 0.093607 | 1 | 0 | 0.928930 | +0.001142 |
| All eligible edges | 182 | 0.894977 | 22 | 1 | **0.949315** | +0.021526 |

**Evidence.** Artifact-verified:
`results/reference_graph_v3/recall_gate_retuned_seed5_max4.json`.

---

## Table 9. Main Structure and Cross-Reference Ablation

**Status:** `REPORT + SLIDES`

**Purpose.** Answer the primary research question by isolating legal structure
and explicit references relative to a strong baseline.

**Protocol.** Use one frozen split, BM25/Harrier/E5-large components, WRRF
weights, component depth, candidate budget, reranker checkpoint, LightGBM
schema, output policy, and evaluator. Define **structure off** as the selected
non-structural token-window representation under the same encoder/token budget;
define **reference on** as graph-v3 same-document expansion with the fixed cap.
Do not reuse separately retuned fusion or selector parameters in individual
cells. Report candidate Recall@50 and nested-OOF final metrics.

| Structure-aware chunks | Same-document references | Candidate Recall@50 | Precision | Recall | Nested-OOF Macro-F2 | Interpretation boundary |
|---|---|---:|---:|---:|---:|---|
| No | No |  |  |  |  | Strong non-structural control |
| Yes | No | 0.904175 |  |  | 0.648354 | Recorded matched control from reference ablation |
| No | Yes |  |  |  |  | Missing controlled cell |
| Yes | Yes | 0.916623 |  |  | 0.647035 | Reference candidate gain; no nested final-F2 gain |

**Evidence.** `docs/research_plan.md`. The first and third cells require new
matched runs; therefore this table is not yet sufficient for a full factorial
claim.

---

## Table 10. Zero-Shot Reranker Input Representation

**Status:** `REPORT ONLY`

**Purpose.** Determine whether the cross-encoder benefits from hierarchical
Khoản–Điểm context and which pair/document token budget is appropriate.

**Protocol.** Freeze the first-stage candidates, zero-shot
`BAAI/bge-reranker-v2-m3`, aggregation, split, candidate pool, evaluator, and
output policy. Change only representation or explicit token budget. Threshold
rows are development diagnostics; fixed top 2 is the policy-neutral comparison.

| Representation / pair-document budget | Fixed-top-2 Macro-F2 | Threshold + fallback-2 Macro-F2 |
|---|---:|---:|
| Điểm, 1024/768 | 0.497310 | 0.548075 |
| Khoản–Điểm, 1024/768 | 0.503088 | **0.558422** |
| Điểm, 1024/512 | 0.499778 | 0.549925 |
| Khoản–Điểm, 1024/512 | 0.503875 | 0.557295 |
| Điểm fallback, 512/448 | 0.501409 | 0.549371 |
| Khoản–Điểm fallback, 512/448 | **0.505002** | 0.555900 |

**Evidence.** `docs/experiments.md`, hierarchical representation ablation.

---

## Table 11. Pointwise MIL Reranker Checkpoint and Pool Comparison

**Status:** `REPORT + SLIDES`

**Purpose.** Select the adaptation epoch without relying on threshold scale and
test whether mining/reranking 100 candidates improves over pool 50.

**Protocol.** Within each pool, reuse the exact checksummed handoff for warm
start and epochs 1–5. Load checkpoints in fresh subprocesses, score identical
candidates, use max chunk-logit article aggregation, and select by fixed top 2.
Threshold/fallback-2 is diagnostic only. Pool comparison holds the graph, WRRF,
mining, optimizer, maximum five-epoch schedule, and seed-cycle sampling fixed.

| Pool | Snapshot | Fixed top-2 Precision | Fixed top-2 Recall | Fixed top-2 Macro-F2 | Pooled threshold Macro-F2 |
|---:|---|---:|---:|---:|---:|
| 50 | Warm start | 0.367580 | 0.615830 | 0.542546 | 0.585978 |
| 50 | Epoch 1 / `1208` | **0.393836** | **0.657610** | **0.579928** | **0.623556** |
| 50 | Epoch 2 / `2418` | 0.381279 | 0.639726 | 0.563353 | 0.616518 |
| 50 | Epoch 3 / `3630` | 0.384703 | 0.641058 | 0.565669 | 0.606536 |
| 50 | Epoch 4 / `4842` | 0.390411 | 0.649429 | 0.573351 | 0.622769 |
| 50 | Epoch 5 / `6054` | 0.391553 | 0.654566 | 0.577044 | 0.621791 |
| 100 | Warm start | 0.368721 | 0.616591 | 0.543516 | 0.587238 |
| 100 | Epoch 1 / `1238` | 0.382420 | **0.639916** | 0.563969 | **0.610202** |
| 100 | Epoch 2 / `2479` | **0.385845** | 0.639612 | **0.565259** | 0.601615 |
| 100 | Epoch 3 / `3721` | 0.382420 | 0.635807 | 0.561410 | 0.601374 |
| 100 | Epoch 4 / `4961` | 0.380137 | 0.630099 | 0.556865 | 0.592438 |
| 100 | Epoch 5 / `6203` | 0.373288 | 0.624201 | 0.550231 | 0.586261 |

**Evidence.** Artifact-verified:
`results/pointwise_mil_pool{50,100}_checkpoint_eval_metrics.tar.gz`.

---

## Table 12. LightGBM Feature-Schema Comparison

**Status:** `REPORT + SLIDES`

**Purpose.** Test whether increasingly rich retrieval, reranker-distribution,
lexical, and static article features improve the frozen neural ranking.

**Protocol.** Freeze pool-50 checkpoint `1208`, 13,348 candidate rows with 522
positives, the same 438 queries, identical query-grouped five folds, seed 42,
LightGBM hyperparameters, and score-gap search. Schemas are nested supersets:
15/41/61/69 features. Fixed top 2 and pooled threshold are diagnostics; nested
score-gap is the primary selector estimate.

| Schema | Features | Fixed top-2 Macro-F2 | Pooled threshold Macro-F2 | Nested-OOF score-gap Macro-F2 |
|---|---:|---:|---:|---:|
| `base_v1` | 15 | 0.583262 | 0.664100 | 0.648388 |
| `enhanced_v1` | 41 | 0.594233 | **0.669570** | 0.649468 |
| `enhanced_v2` | 61 | 0.594233 | 0.663271 | **0.658333** |
| `enhanced_v3` | 69 | **0.597850** | 0.662563 | 0.657139 |

**Evidence.** Artifact-verified:
`results/pool50_epoch1_lightgbm_feature_screen.tar.gz`.

---

## Table 13. LightGBM Feature-Group Removal Ablation

**Status:** `REPORT ONLY`

**Purpose.** Determine which feature families contribute held-out signal and
identify exactly one reduced-schema hypothesis. This is stronger than pruning
from gain importance alone.

**Protocol.** Start from all 61 `enhanced_v2` features. Remove one predefined
group at a time; keep rows, folds, seed, LightGBM parameters, and nested
score-gap protocol fixed. Because many groups reuse the same development split,
interpret positive deltas as exploratory and confirm at most one reduced schema.

| Removed feature group | Nested-OOF Macro-F2 | Change from full 61 features | Confirmatory status |
|---|---:|---:|---|
| None | 0.658333 | 0.000000 | Frozen full schema |
| Retriever agreement | **0.661736** | +0.003403 | Candidate for one confirmation |
| Query/chunk lexical | 0.660375 | +0.002043 | Exploratory only |
| First-stage components | 0.658642 | +0.000309 | Exploratory only |
| Chunk fusion | 0.657927 | -0.000406 | Exploratory only |
| Reference | 0.657108 | -0.001225 | Exploratory only |
| Per-chunk reranker distribution | 0.655033 | -0.003299 | Exploratory only |
| Query shape | 0.653771 | -0.004561 | Exploratory only |
| Chunk length/coverage | 0.651151 | -0.007181 | Exploratory only |
| Within-query calibration | 0.640962 | -0.017371 | Exploratory only |

**Evidence.** Artifact-verified inside the LightGBM feature-screen archive.

---

## Table 14. Final-Set Decision Policy Comparison

**Status:** `REPORT + SLIDES`

**Purpose.** Determine how many articles to return for each query after the
ranking is frozen. This isolates ranking quality from output-cardinality
selection.

**Protocol.** Use exactly the `enhanced_v2` OOF rankings and gold labels for the
438 dev queries. Every learned threshold/calibrator/cardinality model is fit
without its outer validation fold. Fixed top-k requires no tuning. Score-gap
uses its saved nested protocol. Oracle prefix is a non-deployable upper bound.
Report output-size statistics and fold dispersion before selecting a new policy.

| Policy | Precision | Recall | Macro-F2 | Mean output size | Maximum output size | Selection status |
|---|---:|---:|---:|---:|---:|---|
| Fixed top 1 | 0.652968 | 0.565145 | 0.580767 | 1.000 | 1 | Control |
| Fixed top 2 | 0.402968 | 0.674239 | 0.594233 | 2.000 | 2 | Control |
| Fixed top 3 | 0.295282 | 0.730556 | 0.564214 | 3.000 | 3 | Control |
| Nested raw threshold + fallback 2 | 0.499125 | 0.713052 | 0.656754 | 2.235 | 6 | Candidate |
| Nested Platt threshold + fallback 2 | 0.499125 | 0.713052 | 0.656754 | 2.235 | 6 | No extra information over raw |
| Nested isotonic threshold + fallback 2 | 0.499125 | 0.713052 | 0.656754 | 2.235 | 6 | No extra information over raw |
| Nested score gap | **0.532344** | 0.699734 | **0.658333** |  | 3 | Current scientific estimate |
| Logistic gold-cardinality predictor | 0.338128 | 0.732002 | 0.593688 | 3.397 | 5 | Rejected baseline |
| Oracle prefix through k=5 | 0.722793 | **0.794809** | 0.779280 | 1.610 | 5 | Upper bound only |

**Evidence.** Verified local execution of
`scripts/evaluate_decision_policies.py`; score-gap comes from the retained
LightGBM metrics. Monotonic calibration cannot add information when exact
threshold candidates are already searched.

---

## C. Final Results and Presentation Views

## Table 15. Progressive Pipeline Summary

**Status:** `SLIDES ONLY`

**Purpose.** Tell the development story from lexical retrieval to learned
ranking and adaptive output selection.

**Protocol.** This is a summary, not a causal ablation: some rows come from
different screening stages and output policies. Every row must show its split
and policy, and the slide must explicitly say that component-level causal
claims belong to Tables 4, 6, 9, 11, 12, and 14.

| Pipeline stage | Representative configuration | Evaluation policy | Dev Macro-F2 |
|---|---|---|---:|
| BM25 | VnCoreNLP, `k1=1.2`, `b=0.75` | Fixed top 3 | 0.379908 |
| Dense | VietLegal-Harrier | Fixed top 2 | 0.507947 |
| Multi-dense hybrid | BM25 + Harrier + E5-large WRRF | Fixed output from recorded first-stage screen | 0.518868 |
| + Pointwise MIL reranker | Pool 50, epoch 1 / `1208` | Fixed top 2 | 0.579928 |
| + LightGBM | `enhanced_v2`, 61 features | Fixed top 2 | 0.594233 |
| + Adaptive final selection | Same `enhanced_v2` OOF ranking | Nested score gap | 0.658333 |

**Evidence.** Values are recorded or artifact-verified in
`docs/experiments.md` and retained archives. Do not label adjacent differences
as isolated gains on the slide.

---

## Table 16. Official DRiLL Private-Test Context

**Status:** `REPORT + SLIDES`

**Purpose.** Place the project in the official benchmark context without
claiming an uncontrolled head-to-head ablation or tuning on private labels.

**Protocol.** Use only published official private-test results and project
Codabench private results. Preserve each system's own final policy. Mark
external values as published and do not infer missing configuration details or
rank the project retroactively.

| System | Result source | Final policy summary | Private Precision | Private Recall | Private Macro-F2 |
|---|---|---|---:|---:|---:|
| Organizer BM25 | Published external | Unspecified returned k | 0.2265 | 0.3830 | 0.3365 |
| Organizer two-stage | Published external | Fixed top 2 | 0.3740 | 0.6253 | 0.5512 |
| ViDRILL / IUH.TD | Published external | Threshold with fallback | 0.4153 | 0.7605 | 0.6521 |
| CUDO | Published external | Threshold 0.99, top-2 fallback | 0.5097 | 0.7653 | 0.6955 |
| FPT IS | Published external | Threshold/cluster/LLM filter | 0.6222 | 0.7181 | 0.6966 |
| EDM | Published external | Validation-selected LightGBM threshold | 0.6773 | 0.7394 | 0.7261 |
| This project, frozen pool-50 checkpoint-1208 LightGBM | Codabench, submission 1021640 | Threshold + fallback 2 |  |  | **0.6054** |
| This project, same ranking | Codabench, submission 1021639 | Score gap |  |  | 0.6047 |

**Evidence.** `docs/competition_review.md` and `docs/experiments.md`. External
systems are context, not controlled baselines.

---

## Table 17. Public/Private Robustness of Frozen Output Policies

**Status:** `REPORT + SLIDES`

**Purpose.** Test whether the relative ordering of fixed and adaptive policies
is stable across the two hidden splits, without using either split to retune.

**Protocol.** Freeze retrieval, checkpoint `1208`, `enhanced_v2`, LightGBM
training data, and policy parameters before submission. Report submission IDs.
Do not select a new policy from private results.

| Output policy | Public submission | Public Macro-F2 | Private submission | Private Macro-F2 | Public-private difference |
|---|---:|---:|---:|---:|---:|
| Fixed top 2 | 1022898 | 0.5989 | 1022897 | 0.5552 | 0.0437 |
| Threshold + fallback 2 | 1021667 | 0.6524 | 1021640 | **0.6054** | 0.0470 |
| Score gap | 1021732 | **0.6551** | 1021639 | 0.6047 | 0.0504 |

**Evidence.** Codabench results recorded in `docs/experiments.md` and
`docs/final_report_plan.md`.

---

## D. Efficiency, Diagnostics, and Reproducibility

## Table 18. Efficiency and Resource Cost

**Status:** `REPORT ONLY`

**Purpose.** Expose the cost of accuracy improvements and distinguish algorithm
quality from runtime engineering.

**Protocol.** Measure each stage separately after a cold start and a verified
cache hit. Record hardware, software versions, batch size, candidate count,
wall time, throughput, peak process-tree RAM, peak VRAM per device, and artifact
size. Use the same 438-query split where possible. Do not compare local CPU and
Kaggle T4 timings without an environment column.

| Stage / method | Environment | Candidate or pair count | Wall time | Throughput | Peak RAM | Peak VRAM GPU0/GPU1 | Artifact / index size |
|---|---|---:|---:|---:|---:|---:|---:|
| Native BM25, uncached | Local CPU, Python 3.12.9 | top 100 × 438 | 341.36 s |  |  |  | 73.78 MiB |
| bm25s, uncached | Local CPU, Python 3.12.9 | top 100 × 438 | 295.43 s |  |  |  | 55.26 MiB |
| Native BM25, cache hit | Local CPU, Python 3.12.9 | top 100 × 438 | 37.49 s |  |  |  | 73.78 MiB |
| bm25s, cache hit | Local CPU, Python 3.12.9 | top 100 × 438 | 9.16 s |  |  |  | 55.26 MiB |
| Selected multi-dense exact retrieval | Kaggle T4×2 + FAISS CPU |  |  |  |  |  |  |
| Pointwise reranker, pool 50 | Kaggle T4×2, batch 256 |  |  |  |  |  |  |
| Pointwise reranker, pool 100 | Kaggle T4×2, batch 256 |  |  |  |  |  |  |
| LightGBM 61 features | CPU | 13,348 rows |  |  |  | N/A |  |

**Evidence.** BM25 values are recorded in `docs/experiments.md`; remaining
cells require consolidated provenance/log extraction.

---

## Table 19. Diagnostic Subset Performance

**Status:** `REPORT ONLY`

**Purpose.** Test RQ2: whether structure/reference gains concentrate on
multi-article, cross-reference-dependent, long-text, or low-lexical-overlap
queries.

**Protocol.** Define subsets before inspecting method deltas. Gold-cardinality
subsets are deterministic. Explicit-reference subsets must use graph-v3 parsing
or manually audited labels; inferred labels must not be described as official.
Length and lexical-overlap thresholds must be fixed from training/dev
distributions. Evaluate the frozen baseline and treatment on identical qids and
report subset size with macro metrics. Bootstrap paired confidence intervals are
recommended when subset size permits.

| Diagnostic subset | Construction rule required before execution | Queries | Baseline Macro-F2 | Treatment Macro-F2 | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---:|---|
| Single relevant article | `len(unique gold aids) = 1` |  |  |  |  |  |
| Multiple relevant articles | `len(unique gold aids) > 1` |  |  |  |  |  |
| Explicit same-document reference | Audited graph-v3 source mention connected to a candidate/gold |  |  |  |  |  |
| Likely cross-document dependency | Separately audited label; do not infer from graph alone |  |  |  |  |  |
| Long gold article | Predeclared tokenizer-length quantile |  |  |  |  |  |
| High lexical overlap | Predeclared BM25/query-gold overlap quantile |  |  |  |  |  |
| Low lexical overlap | Predeclared BM25/query-gold overlap quantile |  |  |  |  |  |
| Long query | Predeclared query-token quantile |  |  |  |  |  |

**Evidence.** Subsets have not yet been consolidated. Domain-wise analysis is
disabled because the repository lacks an official query-to-domain mapping.

---

## Table 20. Stage-Wise Error Attribution

**Status:** `REPORT + SLIDES`

**Purpose.** Identify whether the remaining failures come from candidate
generation, neural reranking, LightGBM, or final output selection, so that the
next method answers a diagnosed failure rather than adding another model.

**Protocol.** Use the same frozen 438 queries and saved artifacts at every
stage. Assign each query to deterministic, documented categories; report
overlapping and mutually exclusive views separately. For ranking transitions,
compare gold ranks before/after while holding the candidate pool fixed. For
selection errors, define false-positive and missed-gold counts from the final
set. Include paired qids for qualitative audit.

| Error category | Operational definition | Queries | Share | Mean missed gold | Mean false positives | Macro-F2 contribution / loss | Required artifact |
|---|---|---:|---:|---:|---:|---:|---|
| Candidate miss | At least one gold aid absent from first-stage top 50 |  |  |  |  |  | First-stage rankings |
| Reranker demotion | Gold present in pool but moved below the deployable region by reranker |  |  |  |  |  | Chunk candidates + reranker scores |
| LightGBM degradation | Gold rank worsens from reranker aggregation to LightGBM |  |  |  |  |  | Reranker and OOF rankings |
| LightGBM recovery | Gold rank improves from reranker aggregation to LightGBM |  |  |  |  |  | Reranker and OOF rankings |
| Under-selection | Gold is highly ranked but excluded by final policy |  |  |  |  |  | Final-policy predictions |
| Over-selection | Selected set contains avoidable false positives |  |  |  |  |  | Final-policy predictions |
| Truncation exposure | Best gold evidence lies beyond the model input budget |  |  |  |  |  | Source spans + tokenizer audit |

**Evidence.** Required artifacts exist for most stages, but the consolidated
analysis has not yet been executed.

---

## Table 21. Reproducibility and Frozen Configuration

**Status:** `REPORT ONLY`

**Purpose.** Make every main result reproducible and prevent silent drift
between local development, Kaggle evaluation, and submission inference.

**Protocol.** Populate from resolved configs and provenance, not notebook prose.
Record immutable hashes where available. A result is final-report-ready only if
its split, model, artifact, evaluator, and policy can be reconstructed.

| Component | Frozen value | Revision / hash | Artifact or config | Verification status |
|---|---|---|---|---|
| Development split | Seed 42, 20%, 438 qids | Persisted qid list | `split_dev_qids.json` | Verified in retained feature-screen archive |
| Sparse retriever | BM25S/VnCoreNLP, `k1=1.2`, `b=0.75` |  | Resolved experiment config | Dependency/JAR hash still consolidate |
| Chunking | Khoản–Điểm, BGE-M3 tokenizer, 448/64 |  | Shared chunk cache manifest | Hash exists in execution artifacts |
| Dense 1 | `mainguyen9/vietlegal-harrier-0.6b` |  | Dense-cache manifest | Pin revision before next cache generation |
| Dense 2 | `intfloat/multilingual-e5-large` |  | Dense-cache manifest | Pin revision before next cache generation |
| Fusion | WRRF `k=10`, BM25/Harrier/E5 `0.15/0.55/0.30` |  | Frozen handoff/config | Recorded |
| Reference graph | graph v3, same-document, seed top 5, max 4 | Graph/summary hashes | `results/reference_graph_v3/` | Artifact present |
| Reranker | Pool 50 epoch 1 / checkpoint `1208` | Checkpoint hash | Kaggle notebook output | Model files external to Git |
| LightGBM | `enhanced_v2`, 61 features, seed 42, five folds | Feature schema/order | Feature-screen archive | Artifact verified |
| Final policy | Threshold and score-gap frozen controls | Policy parameters in metrics | Submission provenance | Artifact/Codabench verified |
| Evaluator | Macro P/R then F2 | Repository Git revision | `src/drill_ir/evaluation/metrics.py` | Unit-tested |

---

## Recommended Experiment Execution Order

1. **Consolidate missing historical artifacts and provenance.**
   - Dependency: access to archived Kaggle outputs for tokenization, chunking,
     dense screening, and selected multi-dense runs.
   - Config: no new model run; verify resolved config, split qids, predictions,
     metrics, model revisions, and hashes.
   - Output: an artifact manifest and evidence-level upgrades.
   - Updates: Tables 2, 4, 5, 6, 18, and 21.

2. **Run the consolidated stage-wise error analysis.**
   - Dependency: retained selected first-stage, reranker, LightGBM OOF, and
     final-policy predictions.
   - Config: frozen pool-50 checkpoint-1208 `enhanced_v2` pipeline.
   - Output: query-level error JSONL, category summary CSV, audit sample.
   - Updates: Tables 19 and 20.

3. **Confirm exactly one reduced LightGBM schema.**
   - Dependency: frozen 13,348 rows and identical folds.
   - Config: full 61 features versus one predeclared removal of retriever
     agreement; same LightGBM parameters and nested policy evaluation.
   - Output: paired fold metrics and predictions.
   - Updates: Tables 12 and 13.

4. **Develop one information-adding final-set policy.**
   - Dependency: error analysis must show selection rather than ranking is a
     major boundary.
   - Config: frozen `enhanced_v2` OOF rankings; nested query folds; features may
     include early gaps, entropy/slope, component agreement, query shape, and
     legal-reference cues. Do not rerun the neural model.
   - Output: nested predictions, per-fold metrics, output-size distribution,
     oracle comparison.
   - Updates: Table 14.

5. **Run the missing 2×2 structure/reference factorial only if the main research
   claim is required.**
   - Dependency: freeze one non-structural control and one structure-aware
     treatment under the same encoder, WRRF, reranker, LightGBM, and policy.
   - Config: structure off/on × graph-v3 same-document reference off/on.
   - Output: first-stage rankings, nested final metrics, paired deltas, subset
     results.
   - Updates: Tables 8, 9, and 19.

6. **Run one bounded LightGBM hyperparameter sensitivity study.**
   - Dependency: complete policy and reduced-schema decisions first.
   - Config: predeclare a small grid around the current 200 trees, learning
     rate 0.03, 15 leaves, depth 4, min-child 20, L1 0.1, L2 1.0. Use identical
     nested folds and publish every trial.
   - Output: full grid CSV, fold metrics, selected config.
   - Updates: a supplementary sensitivity subsection under Table 12; do not
     create an open-ended model search.

7. **Collect efficiency measurements.**
   - Dependency: frozen final pipeline and clean process boundaries.
   - Config: cold and cache-hit runs on a named local/Kaggle environment.
   - Output: stage timing, pairs/s, RAM, VRAM, artifact sizes.
   - Updates: Table 18.

8. **Final public/private evaluation only for a development-selected method.**
   - Dependency: method and parameters frozen from nested dev evidence.
   - Config: identical inference for public and private, no label-driven edits.
   - Output: submissions, manifests, Codabench IDs and scores.
   - Updates: Tables 16 and 17.

## Core Tables for Final Report

Use the nine consolidated report tables defined at the beginning of this file:

1. **R1 — Dataset and Evaluation Setting.** Compact Table 1 and the essential
   reproducibility fields from Table 21.
2. **R2 — Effect of Legal-Structure-Aware Chunking.** Table 4 unchanged except
   for shorter evidence labels.
3. **R3 — Selection of the First-Stage Retrieval Pipeline.** Retain only the
   sparse anchor, multilingual dense anchor, Harrier winner, and three selected
   dense-pair rows from Tables 2, 5, and 6.
4. **R4 — Structure and Cross-Reference Ablation.** Merge the recall gate from
   Table 8 with the four controlled cells in Table 9.
5. **R5 — Reranker Adaptation and Candidate Budget.** Retain warm start,
   pool-50 epoch 1, pool-50 epoch 5 as a late-epoch control, and pool-100 epoch
   2 from Tables 7 and 11. Move the complete checkpoint sweep to the appendix.
6. **R6 — Learning-to-Rank Features.** Show the four schemas from Table 12 and
   no more than three informative removals from Table 13.
7. **R7 — Final-Set Decision Policies.** Retain fixed top 1/2, raw nested
   threshold, nested score gap, the learned stopping candidate when available,
   and the oracle from Table 14. Omit duplicate Platt/isotonic rows from the
   report and explain their equality in prose.
8. **R8 — Official Benchmark and Hidden-Split Robustness.** Merge Tables 16 and
   17 while clearly separating published external systems from project
   submissions.
9. **R9 — Error Attribution and Cost.** Summarize Tables 18–20; keep detailed
   subset metrics and hardware traces in the appendix.

If the missing cells in Table 9 are not run, narrow the final research claim to
the measured same-document reference recall gate and the existing matched
control/treatment; do not claim a complete structure × reference interaction.

Tables 2, 3, 5, 7, 10, 13, 18, 19, 20, and 21 remain useful evidence catalogue
entries even when their complete forms do not appear in the 15-page report.

## Core Tables for Presentation

Use four result views, normally one per slide:

1. **S1 — First-stage and structure progression.** Compress R2–R3 to at most
   four rows: BM25, dense anchor, selected legal dense model, selected hybrid.
2. **S2 — Main structure/reference result.** Use R4. Until all four factorial
   cells exist, show Table 8 as a reference recall gate and state that final-F2
   evidence is inconclusive.
3. **S3 — Downstream progression.** Compress R5–R7 to reranker, LightGBM, and
   adaptive selection. This is Table 15's storytelling view and must be labelled
   non-causal.
4. **S4 — Final hidden-split result.** Use a reduced R8 with organizer baseline,
   at most two published systems, fixed top 2, threshold, and score gap.

Do not add a separate dataset table slide unless the presentation exceeds 15
minutes. Show dataset scale as large-number callouts. Show error analysis as a
failure funnel or two examples rather than another dense table.
