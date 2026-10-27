# Research plan

## Objective

Build a reproducible platform for retrieval and evidence selection over
Vietnamese legal documents. The current primary research question is:

> RQ1. Under the same retrieval and reranking setup, does explicit modeling of
> legal-document structure and inter-article references improve retrieval
> effectiveness on DRiLL?

Two diagnostic questions refine the claim:

> RQ2. Are any gains concentrated on multi-article or demonstrably
> cross-reference-dependent queries?

> RQ3. What recall improvement and precision loss are introduced by reference
> expansion, and what is their net effect on official macro-F2?

Vietnamese law is the initial domain. The longer-term interest is reliable
retrieval and grounded reasoning over structured, cross-referenced, and evolving
documents.

## Baseline progression

Implement and verify methods in this order:

1. BM25
2. Dense retrieval
3. Hybrid lexical + dense retrieval
4. Hybrid + cross-encoder reranking
5. Structure-aware retrieval
6. Cross-reference-aware retrieval
7. Structure + cross-reference-aware retrieval

Before dense retrieval, verify corpus flattening, article identifiers,
ground-truth mapping, evaluation, and a reproducible BM25 result.

The selected strong pipeline for subsequent controlled comparisons is hybrid
Khoản–Điểm retrieval with 448-token chunks and 64-token overlap, weighted RRF,
then direct reranking of the top 50 fused retrieved chunks with a 512-token
cross-encoder pair budget. Reranker scores are max-aggregated to unique `aid`
values, the thirteen-feature pointwise LightGBM model supplies the final
ranking scores, and query-adaptive early score-gap selection produces the final
set. Article-first rechunking remains an ablation, not the default pipeline.
Hold this configuration fixed when testing structure or reference components.

The completed final-set-selection ablation compared unchanged fixed top-2 and
global-threshold/fallback-2 policies against query-adaptive early score-gap
selection using the same saved LightGBM rankings. Score-gap is now selected for
the strong pipeline because it improved nested-OOF and both reported Codabench
splits. Keep it independently disableable and retain the older policies as
controls. This strengthens the baseline but is not itself evidence for the
structure/reference hypothesis.

## Organizer baselines

The official shared-task overview and its extended journal version report two
baselines:

1. BM25 statistical retrieval.
2. A two-stage system in which BM25 supplies candidates and a cross-encoder is
   fine-tuned with relevant articles and BM25-retrieved negatives using a 5:1
   negative-to-positive ratio and cross-entropy loss. At inference time the
   reranker scores BM25 candidates and returns the final top-k articles.

The extended paper reports that final top-2 is the complete two-stage baseline:
top-1, top-2, and top-5 obtain private-test F2 values of 0.5496, 0.5515, and
0.4615 respectively. Its table defines top-k as the number of returned articles.
It does not disclose the larger BM25 candidate depth passed into reranking. The
BM25-only row also does not explicitly state its returned k, although using top-2
would be consistent with the baseline comparison; treat that as an inference,
not a published configuration fact.

The conference paper reports private-test F2 values of 0.3365 and 0.5512 for
BM25 and two-stage respectively. The journal table gives 0.3368 and 0.5515,
apparently from a slightly different reporting run or rounding. These are
published reference values, not results measured by this repository.

No organizer baseline repository or downloadable configuration was found on the
official task page, Codabench's publicly visible page, either organizer paper,
or targeted GitHub searches as of 2026-09-28. Exact reproduction remains blocked
by unpublished details: BM25 library/tokenization and `k1`/`b`, cross-encoder
checkpoint, BM25 candidate depth, maximum sequence length/truncation, optimizer,
learning rate, batch size, epochs, random seed, and negative-sampling details.

Source: Vuong et al., “DRiLL Shared Task 2025: The Challenge of Deep Retrieval
in the Expansive Legal Landscape,” VLSP 2025,
https://aclanthology.org/2025.vlsp-1.16/.

Extended source: Vuong et al., VNU Journal of Science: Computer Science and
Communication Engineering, 2026,
https://doi.org/10.25073/2588-1086/vnucsce.6463.

Later work may study evidence sufficiency, abstention, temporal/version-aware
retrieval, grounded question answering, and citation evaluation. These are not
part of the initial baseline scope.

## Controlled comparisons

Keep unrelated variables fixed. In particular:

- fusion comparisons use the same dataset, split, embeddings, candidate budget,
  and evaluator;
- reranking comparisons use the same first-stage candidate set;
- reference expansion comparisons change only the reference component relative
  to the strong baseline.

Any multi-component proposal must permit ablation:

```text
Hybrid + Reranker
Hybrid + Reranker + Structure
Hybrid + Reranker + Cross-reference
Hybrid + Reranker + Structure + Cross-reference
```

Do not claim that a component helps unless its contribution can be isolated.

## Dense-model benchmark and ensemble staging

Dense-model diversity is a planned baseline-strengthening treatment before the
structure/reference claims are finalized. It must not become an uncontrolled
model-and-chunking search. The first screening round fixes the existing DRiLL
split, Khoản–Điểm representation, chunk/candidate budgets, exact search, cosine
metric, and a 512-token encoder limit. Set
`retrieval.dense.chunking.tokenizer_model_name` to the selected baseline
tokenizer when the comparison must reuse identical chunk boundaries; otherwise
record that model-specific tokenizers changed both the chunks and embeddings.

The screening registry was narrowed by the project owner on 2026-10-26 to seven
models total. No model has a privileged anchor role: screened dense retrievers
are peers whose contributions are controlled by explicit WRRF weights. Model
identifiers are recorded for reproducible screening; model-card scores and claimed context
lengths are external evidence, not results measured by this repository, and
must be verified before execution.

| Role | Model identifier |
|---|---|
| multilingual baseline | `BAAI/bge-m3` |
| multilingual control | `intfloat/multilingual-e5-large` |
| Vietnamese general | `nhminh107/VietRAG-Embed` |
| Vietnamese legal E5-large adaptation | `mainguyen9/vietlegal-e5` |
| Vietnamese legal instruction model | `mainguyen9/vietlegal-harrier-0.6b` |
| Vietnamese legal E5-base adaptation | `bqbbao6/vietnamese-legal-embedding` |
| Vietnamese legal XLM-R/HaLong adaptation | `truro7/vn-law-embedding` |

Screen every model independently before ensembling. Report dense-only Recall@k
and final reranked macro-F2, runtime, peak memory, embedding dimension, resolved
revision, prefix/instruction, pooling behavior, and whether custom remote code
was required. The primary comparison uses full embedding dimensions; Matryoshka
dimension reduction and context lengths above 512 are later ablations.

The standalone screening control uses `configs/dense_screening.yaml` and
`notebooks/archive/kaggle_dense_standalone_screening.ipynb`. It evaluates exactly one
of the seven registered dense encoders per run, with no BM25, fusion, or
reranker. It requires a verified corpus embedding cache hit and holds seed-42
development IDs, BGE-M3-tokenized Khoản–Điểm 448/64 representations, encoder
limit 512, exact cosine search, candidate depth 100, and native embedding
dimension fixed. Report candidate Recall@1/5/10/20/50/100 and macro-F2 for
fixed output k=1/2/3/4/5/10. Standalone results select plausible ensemble
members; they do not by themselves establish complementarity between any pair.

Do not expand the first screening round beyond these seven models without a
documented scientific reason. Apply each model's verified query/document
prefix or query instruction rather than forcing identical prompt text. Keep the
split, BGE-M3-tokenized Khoản–Điểm 448/64 chunk boundaries, 512-token encoder
limit, exact cosine search, and candidate depth fixed. Screen all seven models
standalone before selecting at most two or three that improve quality and show
useful error complementarity. Do not ensemble all seven by default.

After screening, an ensemble may combine only a small set of complementary
models. Configuration supports named `retrieval.dense_profiles` selected by
`retrieval.hybrid.dense_components`, with an explicit WRRF weight and depth for
each component. Single-dense configurations remain unchanged. Multi-dense runs
persist rankings as `dense:<name>` and fuse exact matching chunks before direct
reranking. The existing thirteen-feature LightGBM treatment expects a single
component named `dense`; it must not be applied to a multi-dense run until
per-model or explicitly rank-normalized dense features are implemented and the
LightGBM model is retrained OOF.

### Session checkpoint: dense screening and low-cost follow-ups

The execution order, updated by the project owner on 2026-10-26 after completion
of the seven-model screen, is:

1. Build and verify one portable cache archive for BM25 and each of the seven
   dense models.
2. Run every dense model standalone with the fixed screening protocol; do not
   use BM25, fusion, or reranking in this comparison.
3. Run the first peer ensemble with `mainguyen9/vietlegal-harrier-0.6b` and
   `intfloat/multilingual-e5-large`, retaining standard BM25 and the strongest
   existing reranker/LightGBM stages. Sweep only the predefined WRRF grid from
   saved component rankings, with the owner-selected BM25/Harrier/E5-large
   ratio `1:5:4` (`0.1/0.5/0.4`) as the primary setting. This is a selected two-model confirmation, not a
   claim that standalone rank alone proves complementarity.
4. Report overlap@k, uniquely recovered gold articles, union Recall@50/100, and
   an explicitly labeled oracle-union diagnostic for the selected pair. Primary
   first-stage selection metrics are Recall@50 and Recall@100; fixed-output F2
   remains secondary.
5. The matched sparse-only screen found BM25+ delta `1.0` worse than standard
   BM25 at Recall@5 through Recall@100 and at best fixed-output F2, although it
   improved rank 1 slightly. Keep standard BM25 in the primary multi-dense run.
   If resources permit, BM25+ may still be run as a secondary end-to-end
   complementarity ablation; keep model pair, split, chunking, depths, reranker,
   LightGBM training procedure, and output policies fixed so its effect remains
   attributable.
6. The predefined 12-case standard-BM25 grid selected the existing
   `k1=1.2`, `b=0.75` configuration by Recall@100, then Recall@50, then fixed
   top-3 F2. Retain it in the primary multi-dense pipeline. The runner-up
   `1.5/0.75` had slightly higher Recall@50 and F2@3 but slightly lower
   Recall@100; do not overinterpret this small dev-only trade-off or report the
   grid winner as an unbiased evaluation result.
7. The local `bm25s` replacement gate passed on 2026-10-27: all ordered
   top-100 rankings for 438 development queries and every reported metric
   matched native standard BM25 exactly. A verified-cache evaluation was 4.09x
   faster and its cache was 25.1% smaller locally. `bm25s` is therefore approved
   as the execution backend for standard BM25. Retain native as the oracle and
   keep native BM25+/RM3 separate because their semantics or support are not
   implementation-equivalent.

Two later inexpensive treatments are allowed after the fixed-weight control:
query-adaptive sparse/dense weights from deterministic query features, and
deterministic reference-edge candidate expansion evaluated first for recall.
Both require independently disableable ablations and query-fold-safe parameter
selection. Do not expand into an unconstrained model/weight search.

### Planned multi-dense resource hardening

The immediate lifecycle repair releases dense models, matrices, query caches,
and full chunk stores after selected chunk candidates are fixed and before the
reranker loads. After confirming that repair on Kaggle, retain the following
implementation order for further reliability work:

1. Persist selected chunk candidates before reranking so a reranker failure can
   resume without repeating retrieval.
2. Split retrieval and reranking into separate processes: retrieval writes
   auditable candidate artifacts and exits; reranking reads only those fixed
   candidates. Process exit, rather than Python garbage collection alone, then
   guarantees reclamation of model, CUDA, multiprocessing, and index state.
3. Persist per-model top-depth chunk rankings with `qid`, component, rank,
   `aid`, chunk identity/text, and score. Reconstruct WRRF candidates offline so
   weight sweeps and selected reranker runs do not reload dense encoders.
4. Reuse one immutable BGE-M3-tokenized Khoản–Điểm chunk store across dense
   models that intentionally share boundaries, instead of constructing and
   retaining duplicate Python chunk objects.
5. Consider normalized memory-mapped corpus embeddings and blockwise exact
   search only after the process boundary is stable; verify that normalization
   does not create full in-memory copies.
6. Record process-tree RSS, CUDA allocated/reserved memory, resident model and
   matrix counts, and stage timings. Add a bounded no-progress failure for model
   loading rather than allowing an apparent stall to consume a Kaggle session.

These are engineering/reproducibility improvements, not new retrieval methods.
Persisted rankings and candidates must be fingerprinted by corpus, split,
chunking, model/revision/prefix, depth, and component name so reuse cannot
silently change an experiment.

The cross-encoder batching optimization is implemented independently of this
resource-hardening list. It scores one deterministic global pair stream in
bounded outer groups and preserves `per_query` as a parity control, without
changing candidates, scores, max aggregation, or tie-breaking. Synthetic
article-rechunk and direct-chunk tests are exact. A matched Kaggle T4 benchmark
of neural batch sizes 8, 16, 24, and 32 is still pending; compare wall time,
pairs/second, peak CUDA memory, and exact output parity, and retain 8 when a
larger batch is unstable. This is an engineering treatment, not a retrieval or
quality treatment.

The first decision gate is the minimal reranker-only notebook comparing batch
8 with 16 on 256 deterministic real legal pairs, adaptive to at least three
repeats and 30 measured seconds per treatment. Its full-run time is an
inference-only projection, not a measured end-to-end runtime. Test 24 and 32
only if 16 is both materially faster and comfortably below the T4 memory limit.

### Deferred participant-inspired methods

The current phase may reuse participant lessons about complementary candidate
generation, small predefined fusion grids, near-positive-aware hard-negative
sampling, and variable-size final selection. These ideas must remain separate
treatments and may not be bundled into the dense-screening baseline.

The following methods are explicitly deferred until the dense screen,
complementarity analysis, one multi-dense confirmation, and the controlled
structure/reference study are complete:

- large-LLM final judges, including Qwen 32B/72B entailment filtering,
  KEEP/DROP classification, pairwise or collective chain-of-thought scoring,
  and multi-round voting or debate;
- a full EDM-style stack combining 500-to-100-to-10 progressive filtering,
  HyDE, generated titles, reference signals, LambdaMART, LightGBM, and LLM
  features;
- HyDE or other generated query-document representations;
- LLM-generated corpus titles or summaries;
- external title/hierarchy reconstruction that requires crawling and uncertain
  `law_id`/`aid` alignment;
- Milvus or another vector database while exact cosine search remains feasible;
- dense-retriever fine-tuning across the screening registry;
- additional reranker families or checkpoint searches before the selected
  epoch-3 control has been evaluated with the chosen multi-dense treatment;
- a new multi-budget dual-level chunking search beyond the existing generic
  and Khoản–Điểm controls; and
- LLM query rewriting or reasoning extensions that were not part of the
  corresponding team's official competition pipeline.

These are deferred rather than rejected. Reopen one only with a new, explicit
research question, a reproducible resource budget, and a comparison that keeps
the other pipeline stages fixed. In particular, deterministic title/hierarchy
enrichment may follow the frozen baseline-v2 comparison when its mapping can be
audited; near-positive-aware hard-negative training may be tested later as one
reranker-supervision ablation; and reference expansion should first demonstrate
candidate-recall value before incurring reranking and final-selection costs.

## Structure and references

Structure-aware and cross-reference-aware retrieval are hypotheses, not assumed
improvements. Potential relations include document hierarchy, explicit provision
references, amendments, replacements, and effective periods.

Every edge must come from explicit document content, deterministic extraction,
or documented annotation. Keep relation extraction separate from retrieval
behavior, and allow structure and reference components to be disabled
independently.

Title enhancement is retained as a later structure experiment, not folded into
the current reranker-recovery run. The first treatment should prepend titles and
hierarchy only when they can be deterministically recovered and mapped to the
supplied `law_id`/`aid`; generated titles are a separate LLM-assisted ablation.
Persist exact, normalized, ambiguous, missing, and manually overridden mapping
statuses, and compare against the unchanged hybrid-plus-reranker candidates.
The published FPT.AI crawl/mapping code is unavailable, so this work requires a
new auditable acquisition and alignment stage before it can be evaluated.

## Error analysis

When needed, generate inspectable analysis tables for failure modes such as
lexical or semantic mismatch, missing relevant articles, multi-article evidence,
cross-reference dependency, reranker demotion, excessive expansion, and false
positives. Aggregate metrics alone are not sufficient for method diagnosis.

Where official or reproducibly derived metadata is available, report overall
metrics together with domain and subset results. Planned subsets include
single-article versus multi-article gold sets and deterministically identified
cross-reference-dependent cases. Subset definitions and query IDs must be
persisted before comparing methods.

Domain-wise analysis is currently blocked because the supplied Parquet files do
not contain organizer domain labels. Do not substitute undocumented model-
generated labels for official annotations.
