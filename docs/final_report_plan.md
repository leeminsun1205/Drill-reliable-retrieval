## Latest discussion and slide addition (2026-11-06)

The owner requested the abstract wording "final evaluation results" and a clear
statement that QA components downstream of retrieval serve demonstration only.
Report 4.4 now offers a qualified explanation of the score gap: missing
LLM-based title/summary enrichment (FPT IS) and relevance features (EDM) may
help representation and candidate discrimination. This is explicitly an
inference to test under controlled comparisons, not a measured cause.
Future work names those components plus LLM evidence filtering, with separate
quality and cost evaluation. No such component was implemented or measured.

The existing fixed-top-2 epoch plot is also inserted as one slide after the
LightGBM feature comparison and before final-set selection. Its short remark
contrasts epoch 3's reranker-only lead with epoch 1's downstream enhanced_v2
lead; no graph values were changed.

## Latest coherence review completed (2026-11-06)

The owner requested a detailed top-to-bottom review. The report now explains
both index preparation and online data flow around the owner pipeline image,
links every stage to its input/output and the next decision, introduces and
interprets all ten tables and three figures, and consistently uses legal
articles, passages and candidate sets as distinct units. The data discussion
motivates adaptive output size; the error discussion distinguishes pair counts
from macro recall and links the largest observed group to future work. Aid
values in examples are explicitly identifiers, not legal article numbers.

The finalized build is 15 A4 pages in Times New Roman, with twelve cited
references and a 264-word abstract. Numeric table cells match the pre-review
version; 22 P/R/F2 rows pass the official formula within rounding tolerance.
All labels, object references and citations resolve, and the LaTeX log is
clean. Layout was reviewed from rendered pages. No experiment/model changes
were made. Current report instructions are consolidated in
`final_report/README.md`; the section and table/figure review map is in
`final_report/review_notes.md`. Earlier status snapshots below are historical.

> **Latest prose cleanup:** the report uses decimal dots and thin-space thousands grouping. Nonessential implementation/library/checkpoint/scheduler details were removed. Article window is explained by 448-token length and 64-token overlap. Selection discussion compares EDM validation-score thresholding with query-specific learned prefix length; benefits are stated as design properties, not superiority over published systems. The bibliography now has 12 cited sources. Experimental configurations and provenance in source-of-truth documents remain unchanged.

> **Cập nhật mới nhất 06/11:** Theo số top-2 chủ đồ án vừa cung cấp, enhanced v2 epoch 1 dùng P/R/F2 = 0.426941/0.701696/0.621680; report/slides hiển thị 0.4269/0.7017/0.6217. Hình epoch so sánh top-2 ở cả hai đường, thay đường threshold trước đó. Ghi chú số cũ phía dưới là lịch sử. Nguồn chi tiết ở docs/experiments.md.

## Latest report state: error analysis completed (2026-11-06)

The 15-page PDF now includes the epoch-F2 plot from the previously recorded
matched checkpoint scoreboard, the measured missing-gold chart (60/97/49),
and cases qid 14663/2406 from the newly uploaded checkpoint-1234 bundle.
All reserved regions are filled. There are ten tables and three figures.
Three error-attribution tests pass; reconstructed outer-fold ensemble scores
match every archived fold and aggregate to 1e-12. See docs/evaluation.md for
full provenance and the outstanding 0.6114 versus 0.6203 top-2 source conflict.
Earlier notes about missing plots/artifacts below are superseded.

# Final experiment and report plan

## Current assembled report (2026-11-06)

The owner requested a serious Vietnamese report aligned with the finalized
slides, within 15 pages. The current `final_report/main.tex` and `main.pdf`
are a 15-page draft including cover, abstract/contents and references, nine
tables and thirteen references. The original member draft remains under
`final_report/original/`. The source URL follows the abstract.

The report directly includes the owner image
`slides/figures/Drill_pipeline_final.png`, intact including its LLM portion.
Table 9 reproduces all ten slide systems in the same order and both score
columns, which the owner confirmed having verified.

Following owner feedback, the report now displays only the slide feature
counts 15/37/57/65. Historical-column and metric-editing notes were removed
from the paper's prose. Original measurement provenance remains documented
in `docs/experiments.md`, `docs/evaluation.md` and `final_report/README.md`;
this is not a newly executed model fit. The corrected enhanced_v2 F2 remains
0.6203. Reference expansion is excluded from the current report scope.

Actual Microsoft Times New Roman regular/bold/italic/bold-italic files are
installed locally under `final_report/fonts/`, loaded directly by XeLaTeX
and ignored by Git. The main text is 12 pt, line spacing 1.15. Font metadata
and Vietnamese glyph coverage were checked; Tinos fallback is removed.

Space is reserved for an epoch-quality plot on page 12 (3.5 cm) and error
analysis on page 14 (9.5 cm), in two files under `final_report/sections/`.
These regions must be filled from current-run artifacts before submission.
No illustrative metrics or fabricated error cases were added. Inspection of
local result bundles found historical checkpoint-1208 evidence, not the
current checkpoint-1234 run; request the current complete-results archive
before attributing errors or drawing a current-run learning curve. Exact
requested artifacts and proposed analyses are in `final_report/README.md`.

This assembly supersedes the historical table-allocation plan below for the
current report. The four research questions follow the final slides:
representation, hybrid retrieval/reranking, signal fusion and output size.
Historical experiments below remain evidence rather than extra requirements
for the current report.

The canonical table-by-table execution blueprint is
`tables_experiment.md`. This file retains the compact finalist summary; when a
status or required measurement differs, the more explicit protocol and evidence
labels in `tables_experiment.md` govern report assembly.

## Frozen finalist

The selected representation is fixed token windows 448/64 with reference
expansion disabled. The frozen neural/retrieval backbone retains pool 50, WRRF
`k=10` with cross-fitted R3 BM25/Harrier/E5 weights `0.15/0.65/0.20`,
pointwise MIL epoch-1 checkpoint `1234` (fixed-window-trained, scheduler
horizon 5), and the 61-feature `enhanced_v2` LightGBM schema. The fixed-window
representation was selected by the R2 dense screen and matched R4 nested-OOF
comparison (which held historical checkpoint `1208` fixed). The current practical submission baseline
uses fixed-window-trained epoch-1 checkpoint `1234`; the checkpoint-1208 run
remains a historical comparison trained on hierarchical article-chunk bags.
The practical baseline designation prioritizes the observed private score and
must be disclosed as post-hoc, not as leakage-free model selection.

Measured official F2 as of 2026-11-02 (new submissions 1027647, 1027650,
1027657, and 1027658):

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
| Fixed 448/64, ref-off, multiclass–Ridge, historical WRRF | 0.6933 | 0.6685 |
| Fixed 448/64, ref-off, multiclass–Ridge, R3-retuned WRRF, checkpoint 1208 | **0.7055** | 0.6735 |
| Fixed 448/64, ref-off, balanced multiclass, checkpoint 1234 | 0.6967 | 0.6809 |
| **Current practical baseline: fixed 448/64, ref-off, multiclass–Ridge, checkpoint 1234** | 0.6978 | **0.6825** |

Relative to LightGBM fixed top 2, learned stopping adds `0.0683/0.0571` on
public/private. It also exceeds score gap by `0.0121/0.0076` and threshold by
`0.0148/0.0069`. Final-set selection is therefore the largest successful
measured treatment after the frozen ranking pipeline. These test results are
external confirmation, not tuning labels.

The listwise treatment nearly matches stopping on public (`-0.0011`) and leads
it on private (`+0.0122`), but the final policy must be chosen from their saved
nested-OOF development evidence rather than this private ordering.

On the historical hierarchical ranking, the closed nested screen selected
balanced multiclass logistic. On fixed windows with historical WRRF weights,
multiclass–Ridge reached `0.6933/0.6685`. The checkpoint-1208 R3-retuned run
reached `0.7055/0.6735` and remains the project's highest measured public score.
The new fixed-window-trained checkpoint-1234 submissions scored `0.6967/0.6809`
for balanced multiclass and `0.6978/0.6825` for multiclass–Ridge. Ridge is the
current practical baseline by private score, while this post-hoc private-based
choice must be disclosed and must not be presented as an unbiased model
selection; scientific selection should follow the saved development protocol.

The retuned Ridge result is recall-oriented: compared with the historical-WRRF
Ridge dev result, precision changes `0.5690 → 0.5449`, recall
`0.7014 → 0.7280`, mean output size `2.176 → 2.358`, and Macro-F2
`0.6702 → 0.6822`. Report this trade-off rather than claiming every stage or
metric improved.

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
selected hybrid retrieval, hybrid plus warm-start cross-encoder, and the current
fixed-window-trained checkpoint `1234` pipeline through LightGBM and
multiclass–Ridge selection. Retain the checkpoint-1208 and fixed-top-2 runs as
historical controls, and compare output policies on matched saved rankings. Published organizer BM25 and
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
