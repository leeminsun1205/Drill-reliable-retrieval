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
values. Article-first rechunking remains an ablation, not the default pipeline.
Hold this configuration fixed when testing structure or reference components.

Before adding a new retriever, a lightweight final-set-selection ablation may
reuse the selected pipeline's saved LightGBM rankings. Compare unchanged fixed
top-2 and global-threshold/fallback-2 policies against query-adaptive early
score-gap selection. Keep it independently disableable, select its parameters
on development data only, and report a nested-OOF selector estimate as well
as pooled tuning. This strengthens the baseline but is not itself evidence for
the structure/reference hypothesis.

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
models total, with `BAAI/bge-m3` as the fixed anchor. Model identifiers are
recorded for reproducible screening; model-card scores and claimed context
lengths are external evidence, not results measured by this repository, and
must be verified before execution.

| Role | Model identifier |
|---|---|
| fixed anchor | `BAAI/bge-m3` |
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
`notebooks/kaggle_dense_standalone_screening.ipynb`. It evaluates exactly one
of the seven registered dense encoders per run, with no BM25, fusion, or
reranker. It requires a verified corpus embedding cache hit and holds seed-42
development IDs, BGE-M3-tokenized Khoản–Điểm 448/64 representations, encoder
limit 512, exact cosine search, candidate depth 100, and native embedding
dimension fixed. Report candidate Recall@1/5/10/20/50/100 and macro-F2 for
fixed output k=1/2/3/4/5/10. Standalone results select plausible challengers;
they do not by themselves establish complementarity with BGE-M3.

Do not expand the first screening round beyond these seven models without a
documented scientific reason. Apply each model's verified query/document
prefix or query instruction rather than forcing identical prompt text. Keep the
split, BGE-M3-tokenized Khoản–Điểm 448/64 chunk boundaries, 512-token encoder
limit, exact cosine search, and candidate depth fixed. Screen all six
challengers standalone before selecting at most two or three that improve both
quality and error complementarity with the BGE-M3 anchor. Do not ensemble all
seven by default.

After screening, an ensemble may combine only a small set of complementary
models. Configuration supports named `retrieval.dense_profiles` selected by
`retrieval.hybrid.dense_components`, with an explicit WRRF weight and depth for
each component. Single-dense configurations remain unchanged. Multi-dense runs
persist rankings as `dense:<name>` and fuse exact matching chunks before direct
reranking. The existing thirteen-feature LightGBM treatment expects a single
component named `dense`; it must not be applied to a multi-dense run until
per-model or explicitly rank-normalized dense features are implemented and the
LightGBM model is retrained OOF.

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
