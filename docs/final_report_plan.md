# Final experiment and report plan

The canonical table-by-table execution blueprint is
`tables_experiment.md`. This file retains the compact finalist summary; when a
status or required measurement differs, the more explicit protocol and evidence
labels in `tables_experiment.md` govern report assembly.

## Frozen finalist

The selected representation is fixed token windows 448/64 with reference
expansion disabled. The frozen neural/retrieval backbone retains pool 50, WRRF
`k=10` with BM25/Harrier/E5 weights `0.15/0.55/0.30`, pointwise MIL epoch-1
checkpoint `1208`, and the 61-feature `enhanced_v2` LightGBM schema. This
representation was selected by the R2 dense screen and matched R4 nested-OOF
comparison, not private feedback. Checkpoint 1208 was trained on hierarchical
article-chunk bags and transfers to fixed chunks at inference; do not describe
it as fixed-window fine-tuning.

Measured official F2 as of 2026-11-02:

| Final policy | Public | Private |
|---|---:|---:|
| Fixed top 2 | 0.5989 | 0.5552 |
| Threshold with fallback 2 | 0.6524 | 0.6054 |
| Score gap | 0.6551 | 0.6047 |
| Fold-safe learned stopping | 0.6672 | 0.6123 |
| Listwise prefix-utility Ridge | 0.6661 | 0.6245 |
| Tuned prefix-utility Ridge (`alpha=1`) | 0.6700 | 0.6265 |
| Balanced multiclass logistic (`C=0.1`) | 0.6847 | 0.6368 |
| Multiclass–Ridge ensemble (`lambda=0.75`) | 0.6863 | 0.6354 |
| Fixed 448/64, ref-off, score gap | 0.6739 | 0.6580 |
| Fixed 448/64, ref-off, balanced multiclass | 0.6833 | 0.6677 |
| Fixed 448/64, ref-off, multiclass–Ridge | **0.6933** | **0.6685** |

Relative to LightGBM fixed top 2, learned stopping adds `0.0683/0.0571` on
public/private. It also exceeds score gap by `0.0121/0.0076` and threshold by
`0.0148/0.0069`. Final-set selection is therefore the largest successful
measured treatment after the frozen ranking pipeline. These test results are
external confirmation, not tuning labels.

The listwise treatment nearly matches stopping on public (`-0.0011`) and leads
it on private (`+0.0122`), but the final policy must be chosen from their saved
nested-OOF development evidence rather than this private ordering.

On the historical hierarchical ranking, the closed nested screen selected
balanced multiclass logistic and that ordering transferred to both hidden
splits. On the selected fixed-window ranking, balanced multiclass reached
`0.6833/0.6677`, while multiclass–Ridge reached the best measured project pair
`0.6933/0.6685`. The final policy label must follow the fixed run's saved
nested-OOF ordering; hidden scores are confirmation rather than a selection
signal.

On the historical hierarchical ranking, the confirmatory ensemble improved
nested-OOF by only `0.000340` and public by
`0.0016`, while private fell `0.0014`; plain balanced multiclass was therefore
the correct historical choice. On fixed windows the interaction changes and
the ensemble leads both hidden scores, so the fixed nested-OOF artifact must
decide whether that gain is reportable as the selected policy. Cost-sensitive
decoding remains a negative result (nested F2 `0.510219`, mean output `4.658`).

## Required report tables

| Table | Rows or factors | Evidence | Status |
|---|---|---|---|
| Dataset and split statistics | Corpus/train/dev/public/private sizes; gold-count distribution; article/chunk lengths | Dataset audit and organizer paper | Mostly available; consolidate |
| Published leaderboard context | Organizer BM25/two-stage and representative teams | `docs/competition_review.md` | Available |
| Baseline progression | BM25; each dense model; hybrid; cross-encoder; LightGBM; final selector | Saved experiment ledger | Consolidate; do not rerun unless a row lacks an artifact |
| Candidate recall | Recall@1/5/10/20/50/100 and pool-50 gold coverage | Frozen retrieval artifacts | Available |
| Structure/reference ablation | no structure/reference; structure only; reference only; both | Frozen first-stage experiments and graph gate | Available, consolidate |
| Dense/fusion ablation | sparse, individual dense models, component combinations, selected WRRF | Multi-dense experiments | Available in Markdown; old local bundles intentionally removed |
| Reranker comparison | warm start; pool-50 epochs 1–5; pool-100 epochs 1–5 | Two checkpoint-evaluation archives | Available |
| LightGBM feature schemas | 15/41/61/69 features under identical folds | Feature-screen archive | Available |
| Feature-group ablation | Remove each interpretable group; gain importance as descriptive evidence | Feature-screen archive | Available; label exploratory |
| LightGBM input depth | 10/20/30/all reranker rows | Feature-screen archive | Available |
| Final policy comparison | fixed k; pooled threshold; nested threshold; score gap; calibrated threshold; cardinality model; oracle | New policy screen | Missing policies to run |
| External result | Dev estimate, public F2, private F2 for frozen submitted policies | Codabench IDs | Available for top-2/threshold/gap |
| Efficiency and resources | Parameters, candidate counts, pairs, runtime, peak VRAM, artifact sizes | Provenance/logs | Partially available; report omissions explicitly |
| Error analysis | Candidate miss; reranker demotion; LightGBM change; under/over-selection; multi-gold/reference subsets | Saved dev rankings | Missing consolidated analysis |

## Baselines and comparison models

The minimum internal baselines are BM25, strongest single dense retriever,
selected hybrid retrieval, hybrid plus warm-start cross-encoder, hybrid plus
fine-tuned checkpoint `1208`, checkpoint `1208` plus LightGBM fixed top 2, and
the same ranking with learned final-set policies. Published organizer BM25 and
two-stage scores are context only, because their implementation is not fully
specified. Published EDM, FPT IS/CUDO, and ViDRILL systems are external context,
not controlled baselines.

## Remaining controlled experiments

1. Run the offline nested decision-policy screen on the frozen `enhanced_v2`
   OOF rankings: raw threshold, Platt-calibrated threshold, isotonic-calibrated
   threshold, score gap, and a small query-cardinality classifier. Keep fixed
   top-k and oracle prefix as controls.

The first verified local execution of this screen (2026-10-31, 438 frozen dev
queries) measured nested F2 `0.656754` for raw, Platt-calibrated, and
isotonic-calibrated thresholds, versus the already saved nested score-gap F2
`0.658333`. The logistic cardinality baseline measured `0.593688`; fixed top 2
measured `0.594233`; the non-deployable oracle prefix at maximum k=5 measured
`0.779280`. Monotonic calibration did not improve exact threshold selection,
so subsequent policy work should add decision information rather than merely
rescale the same score.
2. Select at most one policy by nested-OOF F2, fit it on all OOF development
   predictions, and package public/private ZIPs from already saved test
   LightGBM predictions. Do not rerank.
3. If policy work is complete, run one bounded LightGBM grid under the same
   query folds and nested selector protocol. Predeclare the grid; report every
   trial. Do not use public/private feedback for selection.
4. Produce the consolidated error-analysis and final tables. Stop model search
   unless the audit identifies a specific failure boundary.

## Decision rule

The new policy replaces score gap only if its nested-OOF gain is positive and
stable across outer folds, with no pathological output-size tail. Public and
private results are reported afterward and cannot reverse the development-time
selection. If no policy satisfies this rule, retain score gap as primary and
threshold as the robustness control.
