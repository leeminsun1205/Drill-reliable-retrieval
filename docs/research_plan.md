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

The selected representation is overlapping fixed token windows with a
448-token document budget and 64-token overlap, with reference expansion
disabled. The current practical submission pipeline uses VnCoreNLP BM25,
VietLegal-Harrier, and multilingual-E5-large fused by WRRF `k=10` with weights
`0.15/0.65/0.20`; it reranks the pool-50 fused chunks with the pointwise-MIL
cross-encoder (pair length 512), max-aggregates chunk scores to unique `aid`
values, builds the 61-feature `enhanced_v2` LightGBM scores, and selects the
final set with the multiclass–Ridge policy. Its reranker is fixed-window-trained
epoch 1 / checkpoint `1234` (three adaptation epochs on a five-epoch scheduler
horizon). The prior checkpoint-1208 run used hierarchical article-chunk bags
and is a historical cross-representation-transfer result, not the current
reranker.

The owner-reported checkpoint-1234 multiclass–Ridge submissions scored
`0.6978` public and `0.6825` private (IDs `1027650/1027647`); this is the
current practical baseline because private performance is prioritized. The
checkpoint-1208 run remains the public record at `0.7055`. Selecting the
practical baseline after observing private scores is a post-hoc operational
choice, not leakage-free scientific model selection. Do not continue tuning on
private feedback; use saved development evidence for methodological claims.
Hierarchical chunks and references remain controlled treatments, not parts of
the current pipeline.

The completed matched Table-R4 factorial, which holds the earlier
checkpoint-1208 reranker fixed, supplies the representation/reference ablation
evidence:
fixed 448/64 windows achieved nested-OOF F2 `0.6681/0.6669` with references
off/on, versus `0.6597/0.6583` for hierarchical Khoản–Điểm. Same-document
references increased candidate Recall@50 in both representations but slightly
reduced final F2; the F2 difference-in-differences was about `0.0002`. Thus the
point estimates do not support a structure/reference interaction and identify
fixed windows as the stronger development control. The policy-matched
fixed-window rerun subsequently retained public performance and improved
private performance, but those hidden scores are confirmation rather than the
selection signal. Historical Khoản–Điểm/reference-on results remain valid
ablations. Artifact/hash verification and paired uncertainty are still pending.

The completed multi-dense treatment improves the user-reported matched
threshold submission from single-BGE public/private F2 `0.6160/0.5882` to
`0.6384/0.5952`. Its nested-OOF score-gap estimate is `0.660836`, but the frozen
multi-dense threshold submission outscored score-gap on both reported
leaderboard splits (`0.6384/0.5952` versus `0.6260/0.5920`). Record this policy
reversal as test-distribution evidence; do not use the private result to tune a
replacement policy.

The completed final-set-selection ablation compared unchanged fixed top-2 and
global-threshold/fallback-2 policies against query-adaptive early score-gap
selection using the same saved LightGBM rankings. Score-gap improved nested OOF
and earlier single-dense submissions, but the multi-dense threshold policy led
on both newly reported leaderboard splits. Keep both independently reproducible
and do not use private feedback to choose another policy. Final-set selection
strengthens the baseline but is not itself evidence for the
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

The presentation's future-work slides prioritize limitations of the current
pipeline: incomplete document-level legal metadata; semantic mismatch between
natural-language queries and formal provisions; the cross-encoder's 512-token
pair limit; and retrieval signals not yet enriched with costly LLM-derived
features. Addressing these limitations could improve retrieval, but gains are
not established and full LLM-enriched cascades may exceed the available GPU,
runtime, and API budget. Follow-up work should first enrich law/article titles,
hierarchy, and version metadata from authoritative legal sources with
identifier validation and provenance. Then compare query expansion and
auxiliary article summaries while retaining the original text. Longer-context
rerankers or controlled position-embedding extensions require matched-candidate
quality and compute comparisons. As external DRiLL methods, EDM reports
generated titles, LLM-derived features, and cascaded LambdaMART/LightGBM;
FPT IS reports LLM data augmentation and hierarchical chunking; ViDRILL reports
hard-negative mining. These are ideas to test, not results of this project.
Start with a bounded hard-negative experiment, then evaluate cached enrichment
on the frozen top-50 candidate set and report its generation and inference
costs. Pilot the bounded training-side treatment first. Do not use private-test
scores to select among these treatments.

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

## Historical architecture checkpoint after the multi-dense baseline

This 2026-10-27 snapshot records the earlier active strong pipeline; the current
submission configuration is summarized at the start of this document:

```text
legal corpus
  -> shared BGE-M3-tokenized Khoản–Điểm 448/64 chunks
  -> BM25S + VietLegal-Harrier + multilingual-E5-large
  -> weighted RRF 0.1 / 0.5 / 0.4
  -> top-50 fused chunks
  -> epoch-3 cross-encoder, pair length 512, global-stream (batch 64 for the
     active T4x2 lifecycle verification; batch 128 remains the isolated winner)
  -> max score aggregation to aid
  -> query-grouped OOF LightGBM with an explicit feature schema: historical
     base_v1 (15), enhanced_v1 (41), enhanced_v2 (61), or enhanced_v3 (69)
  -> threshold/fallback or query-adaptive score-gap final selection
  -> official qid/relevant_laws submission
```

This architecture has completed the main engineering stabilization work:

- BM25S exactly matched native BM25 over all 43,800 development top-100
  positions and was faster on a verified cache hit;
- all seven screened dense corpus embedding caches remain reusable;
- Harrier and multilingual-E5 share one checksummed deterministic chunk cache;
- dense matrices load read-only through memory mapping and exact cosine search
  uses FAISS Flat rather than an approximate index;
- dense encoders, matrices, query caches, chunk objects, and search indexes are
  released before the cross-encoder loads;
- cross-encoder pairs are scored as one bounded global stream; the isolated
  two-T4 benchmark selected batch 128 with about 3.08 times batch-8 throughput
  and sampled top-1/top-3 parity, while the current end-to-end lifecycle
  verification conservatively uses batch 64; and
- LightGBM training and evaluation remain query-grouped, with nested OOF used
  when estimating an adaptive final-set policy.

The remaining system bottlenecks are orchestration rather than model compute.
Implement them in this order when more reliability work is required:

1. Persist the exact selected top-50 chunk candidates, including query, chunk
   text/identity, parent `aid`, component ranks/scores, and fused score, before
   the reranker starts.
2. Run retrieval and reranking in separate processes. Retrieval writes an
   immutable, fingerprinted candidate artifact and exits; reranking consumes
   only that artifact. A reranker failure must not repeat retrieval.
3. Persist per-model top-depth chunk rankings so WRRF weights, component
   ablations, and selected candidate reconstruction can run offline without
   loading dense models or rebuilding FAISS indexes.
4. Record stage wall time, pairs/second, process-tree RSS, peak CUDA
   allocated/reserved memory, and resident model/matrix counts in provenance.
5. Pin exact Hugging Face model and tokenizer revisions when producing the next
   cache generation. Current caches remain usable, but a future cache must not
   pair old corpus vectors with query encoding from a silently updated model.

Do not prioritize query-embedding caches, serialized FAISS Flat indexes,
float16 corpus caches, ANN/HNSW/IVF search, or reranking 100 candidates. Query
encoding and exact-index construction are not the dominant costs; float16 and
ANN introduce ranking risk; and moving from top 50 to 100 doubles cross-encoder
pairs for only about `0.02957` additional first-stage macro recall in the
measured multi-dense run.

### Current F2 diagnosis

The main quality bottleneck is after candidate generation. Multi-dense
candidate Recall@50 is about `0.92208`, whereas the measured LightGBM threshold
recall is `0.73752` and pooled score-gap recall is `0.69460`. The deployable
nested-OOF score-gap estimate is F2 `0.660836`, while the non-deployable
best-prefix-through-5 oracle is F2 `0.767165`. This gap indicates substantially
more headroom in within-pool ranking and output-cardinality selection than in
simply retrieving deeper candidates.

The multi-dense final-policy ordering is also distribution-sensitive. Pooled
development score-gap F2 (`0.663198`) exceeds threshold/fallback-2
(`0.647531`), but the reported threshold submissions exceed score-gap on both
public (`0.6384` versus `0.6260`) and private (`0.5952` versus `0.5920`). Keep
nested OOF as the scientific estimate, report frozen leaderboard policies
separately, and never use the private reversal to tune another selector.

Before introducing another model, produce a query-level error audit with these
mutually interpretable categories:

- gold absent from the top-50 candidate pool;
- gold present but demoted by the cross-encoder;
- cross-encoder ordering improved or damaged by LightGBM;
- ranking adequate but final selection returned too few articles;
- final selection returned avoidable false positives;
- multi-gold queries; and
- queries containing explicit or likely cross-provision dependencies.

Report counts, macro-F2 contribution, output-size distributions, and paired
query IDs. This audit decides whether the next treatment belongs in retrieval,
reranking, learned fusion, or final selection.

### Prioritized F2 roadmap

1. **Final-set calibration and cardinality.** On the same saved LightGBM
   rankings, test fold-safe Platt or isotonic calibration and a small predictor
   for output cardinality `k in {1,2,3}`. Candidate features may include early
   score gaps, score entropy/slope, component agreement, legal-citation cues,
   query length, and multi-provision language. Evaluate every choice with
   nested query folds; the pooled development optimum is diagnostic only.
   The first implemented information-adding treatment is a sequential logistic
   continue/stop classifier through `k=5`. Its outer-fold estimate nests
   threshold selection through inner cross-fitted probabilities, and its
   deployment model is fit only after the nested estimate is recorded. This is
   an executable pending experiment, not a measured improvement.
2. **Deterministic reference-expansion recall gate.** Extract only explicit,
   auditable provision references; expand from fixed baseline candidates; and
   first measure newly recovered gold articles, Recall@50, precision loss, and
   affected query subsets. Send expanded candidates to the expensive reranker
   only if the recall gate succeeds.
3. **Deterministic structure enrichment.** Add recoverable law/article titles
   and hierarchy paths (`law -> chapter -> section -> article -> clause ->
   point`) while keeping candidates, models, budgets, and final policy fixed.
   Generated titles are a separate later ablation.
4. **Reranker supervision.** If error analysis shows gold candidates are being
   demoted, mine hard negatives from the frozen multi-dense top-50 flow, label
   only evidence-bearing gold chunks as positives, and compare pointwise with
   pairwise/listwise objectives. Near-positive same-law or same-article chunks
   must be treated explicitly rather than silently mislabeled.

The active 2026-10-30 reranker treatment is FPT-inspired article-labelled
supervision over frozen Khoản–Điểm chunk bags. During each training epoch, the
next non-overlapping slice of at most four chunks is selected from a
deterministically shuffled cycle for each article and positive/negative field;
inference still scores every frozen candidate chunk and aggregates each article
by maximum chunk logit. This bounded representation retains article-level gold
labels, covers a bag before repeating it, and avoids unbounded all-chunk DDP
steps; the permanent mining rows continue to contain every chunk.
For each non-gold `aid`, mining retains only its maximum-teacher-score chunk,
takes the top 30 unique articles, drops the five highest-ranked uncertain
articles, and deterministically samples 10 negatives when more than 15 remain
or 5 otherwise. Every gold `aid` is a positive bag containing its exposed
chunks, or deterministic same-chunker fallback chunks when absent; the teacher
does not select positives. Pointwise BCE max-pools within the selected slice.
Pool 50 and pool 100 form the active controlled candidate-budget comparison;
every run trains five epochs but model selection must use the matched
development evaluation for every checkpoint.

Pairwise pool-50 and pool-100 training is deferred after repeated observed DDP
stalls. This is an execution-reliability decision, not a measured conclusion
about pairwise quality. Before revival, the separate positive/negative DDP
forwards must be replaced by one combined forward and validated in an isolated
multi-GPU smoke test.

Development retrieval is frozen independently from fine-tuning and checkpoint
evaluation. `notebooks/archive/preparation/kaggle_prepare_frozen_dev_handoff_pool50.ipynb` and
`notebooks/archive/preparation/kaggle_prepare_frozen_dev_handoff_pool100.ipynb` create the two
reference-on handoffs without loading a reranker. Each archive contains the
handoff, resolved configuration, and a SHA-256 manifest covering exactly the
persisted 438-query development split. Every pointwise checkpoint at a given
pool size must reuse the matching handoff; this is an execution split, not a
methodological change.

The execution order is frozen as follows: finish both pointwise runs; evaluate
the warm start and all five checkpoints on their matching handoffs; select the
pool/checkpoint finalist by development fixed-top-2 F2 with fixed top 1/3 and
cost as required diagnostics; and only then train the LightGBM selector. Do not
train LightGBM independently for every checkpoint,
because doing so would confound neural-checkpoint selection with selector
capacity and multiply development selection. The final selector compares
`base_v1` against the enhanced schemas with group-removal ablations for
normalization, retriever agreement, reference provenance, per-chunk reranker
evidence, length/lexical evidence, article-static features, chunk statistics,
and query cues. Feature importance is descriptive only: remove a
feature group only when nested query-fold OOF shows no material F2 loss and no
meaningful increase in fold variance. Public/private test scores must not choose
the feature set, input depth, threshold, or score-gap policy.
5. **Query-adaptive fusion.** After the fixed-weight baseline, predict a small
   sparse/dense weight policy from deterministic query and component-agreement
   features using query-fold-safe tuning. Do not reopen an unconstrained weight
   grid.
6. **Groupwise learned ranking.** Compare the current pointwise LightGBM model
   with a LambdaRank/LambdaMART treatment using the same 41 features, `qid`
   groups, candidates, folds, and final selector. This is a later ranking
   ablation, not a simultaneous architecture replacement.

The mandatory controlled matrix remains baseline, structure only, references
only, and structure plus references. The engineering baseline is now strong
enough that the next scientific priority is query-level error analysis followed
by the reference and structure treatments, not another broad model search.

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
`notebooks/archive/retrieval/kaggle_dense_standalone_screening.ipynb`. It evaluates exactly one
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
   retaining duplicate Python chunk objects. This item is implemented with a
   checksummed shared chunk cache; a full-corpus cache-hit run remains the
   operational confirmation.
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
of neural batch sizes 8, 64, and 128 is complete; it compared wall time,
pairs/second, peak CUDA memory, and sampled ranking parity and selected batch
128. This is an engineering treatment, not a retrieval or quality treatment.

The first decision gate is the reranker-only notebook comparing batch 8, 64,
and 128 on 500 deterministic real legal pairs arranged as ten query-specific
top-50 proxy groups, adaptive to at least four
repeats, 2% early-stop CV, and 60 measured seconds per treatment when noisy.
Its full-run time is an inference-only projection, not a measured end-to-end
runtime. Select the fastest treatment only when sampled ranking parity and T4
memory-headroom checks pass against batch 8.

The 2026-10-27 two-T4 decision-gate run selected batch 128: it delivered about
89.37 pairs/second versus 29.03 for batch 8 (3.08x), used 3,659 MiB peak
allocated memory on the most-used GPU, and preserved sampled top-1 and top-3
rankings exactly. Treat this as measured support for the next full-run
configuration, not proof of parity for every development query.

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

The implemented query-level retrieval audit is the first diagnostic gate for
the current multi-dense baseline. It must report per-query Recall@50 deficit,
component complementarity, fusion exclusions, actual unique-article reranker
coverage, and later-stage demotions before another retriever is introduced.
Component names are discovered from artifacts rather than fixed to the current
Harrier/E5 pair, preserving the same analysis for future dense replacements.

### Session checkpoint: dense-pair selection and next optimization gate

As of 2026-10-28, the immediate decision is among three matched no-expansion
multi-dense treatments, each anchored by VietLegal-Harrier and differing only
in its second dense peer: BGE-M3, VietLegal-E5, or multilingual-E5-large. The
corresponding Kaggle notebooks are
`kaggle_multi_dense_harrier_bge.ipynb`,
`kaggle_multi_dense_harrier_vietlegal_e5.ipynb`, and
`kaggle_multi_dense_harrier_e5_retuned.ipynb`. All retain the seed-42 split,
BM25S/VnCoreNLP, BGE-M3-tokenized Khoản–Điểm 448/64 chunk store, `top_chunks`,
the 40-case WRRF screen, epoch-3 cross-encoder, packed-stream batch 256,
nested-OOF LightGBM evaluation, and disabled neighbor/reference expansion.
Results are pending and must not be inferred from standalone dense ranking.

Select the dense pair using candidate Recall@20/50/100, paired per-query
wins/losses, unique recovered gold, oracle-union recall, actual fixed-budget
fusion recall, reranker results, nested-OOF LightGBM, and runtime/memory. Do not
select solely on the pooled development F2 or standalone dense F2. Model-family
similarity is a hypothesis only: VietLegal-E5 and multilingual-E5-large share
an E5-large lineage, while VietLegal-Harrier and VietLegal-E5 used substantially
overlapping legal training sources despite different base models. Empirical
ranking overlap and unique-gold recovery decide complementarity.

After selecting the pair, freeze it before deeper retrieval tuning. First run a
coarse-to-fine WRRF search around the winning `rrf_k` and weights, then consider
component-specific depths and candidate budgets 50/75/100. The next independent
treatments, in order, are fixed-WRRF explicit-reference expansion, deterministic
citation/query normalization, conservative same-law radius-one expansion, and
only then metadata features or OOF learned fusion. At the current roughly
92--93% development Recall@50 level, expect diminishing returns; preserve a
fixed candidate budget and prioritize paired recovery over unbounded oracle
union gains.

The retuned no-expansion Harrier + multilingual-E5 run is the control for the
reference-enabled main notebook. If reference expansion changes the selected
WRRF case, report both the end-to-end retuned treatment and a fixed-WRRF
reference-only ablation. Never label their combined difference as the isolated
graph effect.

Only after the winning first stage and reference decision are frozen should a
new reranker be trained. Persist the exact component rankings, fused candidates,
chunk candidates, split IDs, resolved config, and cache fingerprints. Mine
training-fold hard negatives from that frozen candidate distribution, retain
all useful positive chunks for gold articles, and compare the new reranker with
epoch 3 on exactly the same candidates. Keep reranker supervision, LightGBM,
and final-selection policy as separate ablations, and reserve untouched or
nested evaluation to control accumulated development tuning.

### Measured reference-expansion result on the retuned dense control

The 2026-10-28 matched seed-42 development comparison used the selected
Harrier + multilingual-E5-large control with `rrf_k=10`, BM25/Harrier/E5
weights `0.15/0.55/0.30`, `top_chunks`, candidate budget 50, and the epoch-3
reranker. The treatment added deterministic same-document explicit-reference
expansion from the top five seeds, with at most four available-chunk
expansions and a fixed candidate budget. The saved split IDs, component
rankings, and fused first-stage predictions are byte-identical between the two
artifacts, so this is a fixed-WRRF reference-only ablation rather than a
retuned end-to-end comparison.

Reference expansion changed the reranker chunk ranking for 110/438 queries.
Candidate Recall@50 increased from `0.904175` to `0.916623` (`+0.012448`): 11
queries gained gold coverage and none lost it. Gains occurred only after rank
20; Recall@1/5/10/20 was unchanged at the candidate-construction stage. The
epoch-3 reranker retained the candidate gain at depth 50, but fixed top-2 F2
was unchanged at `0.533073`; its threshold-policy F2 increased only from
`0.598183` to `0.598525`.

Fresh LightGBM OOF training improved fixed top-2 F2 from `0.562469` to
`0.580849`, but deployable-policy evidence was mixed. Pooled score-gap F2 moved
from `0.650374` to `0.651078`, while the leakage-controlled nested-OOF estimate
moved from `0.648354` to `0.647035`. Treat the reference component as a real
candidate-recall improvement but not yet a demonstrated final-selection F2
improvement. The next controlled diagnostic should apply the control-trained
reranker/selector unchanged to both candidate sets, or retrain a reranker on
the frozen reference-expanded candidate distribution and evaluate it on an
untouched split; do not select the graph treatment from pooled development F2.

The subsequent graph-v3 freeze audit inspected all 12 recovered gold aids
across those 11 queries. Every recovery had an explicit same-document source
mention whose target was the recorded gold provision. The audit also found
compact coordinated Điều lists omitted by v2 and arbitrary edge-file ordering
when more than four targets were reachable. Graph v3 adds deterministic compact
integer-list extraction, excludes self-loops from the candidate graph, and
selects at most four available reference targets by their existing base chunk
rank. Alphanumeric Điều identifiers and ambiguous/named-without-ID references
remain fail-closed. Use graph v3, same-document scope only, for the final
candidate-budget matrix; cross-document expansion remains a separate low-signal
treatment.

Freeze this graph definition after the audit. On the same 438-query
retrospective gate, v3 expanded 172 rather than 170 queries under the
same-document policy, but both v2 and v3 recovered 21 fixed-budget gold pairs,
dropped one, and produced identical Recall@50 (`0.948174`) in this
article-level upper diagnostic. Cross-document v3 expanded 24 queries and
recovered only one gold pair. Further parser broadening is therefore outside
the current first-stage study unless a concrete failed case or independent
annotation demonstrates a missing edge that can be resolved without guessing.
This freeze denotes a conservative, reproducible experimental graph; it is not
a claim that the supplied corpus has been converted into a complete or
authoritatively annotated legal citation graph.

### Measured three-pair dense selection

The matched seed-42 three-pair screen completed on 2026-10-28 for Harrier
paired with multilingual-E5-large, VietLegal-E5, or BGE-M3. Each treatment used
the same BM25S/VnCoreNLP component, chunk store, 40-case WRRF screen,
`top_chunks` candidate construction, epoch-3 reranker, and fresh five-fold OOF
LightGBM. The winning WRRF settings were allowed to differ within that shared
predeclared grid, so these values select an end-to-end dense pair and are not a
fixed-weight encoder-only ablation.

Harrier + multilingual-E5-large achieved fused Recall@20/50/100 of
`0.875826/0.927789/0.952794` and nested-OOF score-gap F2 `0.648354`.
Harrier + VietLegal-E5 improved Recall@50 to `0.933823` but reduced Recall@1
from `0.488965` to `0.431126`, was essentially tied at Recall@100 (`0.952033`),
and reduced nested-OOF F2 to `0.630950`. Paired against multilingual-E5-large,
it won/lost 11/4 queries at Recall@50 but only 4/3 at Recall@100, while losing
37 and winning 10 at Recall@1.

Harrier + BGE-M3 achieved fused Recall@20/50/100 of
`0.878164/0.926593/0.949369` and nested-OOF F2 `0.629978`. Against
multilingual-E5-large it won/lost only 6/3 queries at Recall@50 and 1/2 at
Recall@100; its candidate Recall@50 (`0.902653`) was also slightly below the
multilingual-E5-large control (`0.904175`). Its incomplete public/private
submission run is irrelevant to this selection because first-stage retrieval,
development reranking, OOF LightGBM, and query analysis all completed.

Freeze Harrier + multilingual-E5-large as the selected dense pair. VietLegal-E5
provides a small depth-50 coverage gain but does not survive the preregistered
downstream and nested-OOF criteria; BGE-M3 provides no material coverage or
final-selection advantage and has a less reliable Kaggle execution path.
Freeze the primary first-stage WRRF at `rrf_k=10` with normalized
BM25/Harrier/multilingual-E5-large weights `0.15/0.55/0.30`, component depth
100, and candidate budget 50. The 40-case screen showed that the leading
Recall@50 cases cluster at `rrf_k` 10--20 and that further fine-grained tuning
on the same 438-query development split offers little value relative to its
selection-bias risk. Do not reopen encoder or WRRF search while evaluating
reference and structure treatments.

### Next execution gate: candidate budget by reference expansion

Before retraining the reranker, run one matched 2-by-2 gate with candidate
budgets 50 and 100 crossed with deterministic same-document explicit-reference
expansion disabled and enabled. Keep BM25, both dense encoders, chunking,
component depth 100, WRRF, split IDs, epoch-3 reranker, output cutoffs, and the
control threshold fixed. Reuse saved component rankings and caches; do not
re-encode the corpus or reopen fusion tuning.

For each cell report candidate Recall@20/50/100 where defined, paired per-query
wins/losses, recovered and displaced gold aids, epoch-3 top-1/2/3 and fixed-
threshold Precision/Recall/F2, reranked pair count, wall time, and peak memory.
Fresh LightGBM OOF results may be reported as a downstream end-to-end
diagnostic, but must not replace the fixed-reranker/fixed-policy comparison when
attributing an effect to reference expansion or candidate budget. Audit every
query whose gold coverage changes under reference expansion before accepting
the graph treatment.

Select the candidate distribution by end-to-end quality under measured cost,
not Recall@k alone. Budget 100 must improve final performance enough to justify
roughly doubling reranker pairs; otherwise retain budget 50. Reference expansion
must use the same edge rule and expansion cap in both budgets. After this gate,
freeze the winning candidate distribution, mine its hard negatives, and train
the next reranker. Do not run further dense-model, BM25, WRRF, HyDE, generated-
title, external-crawl, or large-LLM experiments before that reranker comparison.

The active post-freeze reranker comparison is pointwise BCE at candidate pools
50 and 100. Both runs warm-start from the attached
`AITeamVN/Vietnamese_Reranker` weights and hold the graph, WRRF, mining rule,
optimization, and evaluation protocol fixed, so the comparison isolates the
candidate budget. Mining uses the same teacher and frozen candidate rule: for each
gold aid, select the teacher-best chunk among gold chunks actually exposed to
the reranker, falling back to same-chunker representations from the full gold
article only when that aid is absent. Rank non-gold candidate chunks, exclude
the five most suspicious of the top 30, and deterministically sample ten
semi-hard negatives. Both pool treatments exclude the persisted 438 development
qids and compare fixed top-1/2/3 on identical development candidates before
LightGBM is used for downstream diagnosis. Pairwise remains a documented future
objective, but is outside this active gate until its multi-GPU execution path is
reworked and passes an isolated smoke test. Epoch 3 remains the established
task-fine-tuned control.

The attached Vietnamese checkpoint contains inference weights, not optimizer,
scheduler, scaler, or RNG state. These treatments are therefore weights-only
warm starts with a fresh five-epoch schedule. Their snapshots are named
adaptation epochs 1--5 and must not be described as continuation epochs. Five is
a maximum checkpoint horizon rather than an assumption that the last snapshot
will win; retain and evaluate every epoch. Record the attached safetensors
SHA-256 before mining, and retain exact-resume state only for the latest new
adaptation checkpoint while keeping inference weights for every epoch.

Scoring semantics remain deliberately simple. Pointwise training applies
binary cross-entropy directly to the model's single raw logit with targets one
and zero; no sigmoid is required for ranking. Inference keeps the existing
maximum chunk logit per `aid`. Because adaptation can shift the absolute logit
scale, fixed top 2 is the
primary checkpoint gate, fixed top 1/3 are required diagnostics, and every
threshold is retuned on development per checkpoint rather than inherited from
the Vietnamese or epoch-3 controls.

### Final first-stage gate: pool 50 with same-document references

The four 2026-10-28 pool-50/100 by reference-off/on artifacts passed the
matched-run audit:
their development split, component rankings, and fused article predictions are
byte-identical, and their resolved configurations differ only in the declared
pool/reference treatment.

Increasing the chunk-pair pool from 50 to 100 without references raised actual
candidate recall from `0.904175` to `0.941433`, recovered 30 gold pairs across
28 queries, dropped none, and raised complete-gold queries from 369 to 392.
However, the unchanged epoch-3 reranker became worse at every principal fixed
cutoff: top-1/2/3 F2 changed from `0.514270/0.533073/0.518575` to
`0.493488/0.512372/0.495090`. At top 2, pool 100 won/lost 9/21 queries in
recall. A paired query bootstrap for the top-2 F2 difference gave `-0.020701`
with a 95% interval of `[-0.038733, -0.003241]`. Nested-OOF score-gap F2 also
fell from `0.648354` to `0.639096`. The pair count doubled from 21,900 to
43,800, while measured development retrieval plus reranking time increased
from 611.5 to 871.8 seconds. These results reject pool 100
under the predeclared fixed-reranker-and-cost rule despite its larger recall
ceiling.

At pool 100, same-document graph v3 changed candidate sets and reranker rankings
for 49/438 queries but recovered and dropped no gold candidate pair. Epoch-3
top-1 and top-2 were exactly unchanged; top-3 lost gold on two queries and F2
moved from `0.495090` to `0.492344`. Nested-OOF score-gap F2 moved from
`0.639096` to `0.637982`. This is evidence that references are redundant at
pool 100, not a decision about their pool-50 treatment.

At pool 50, enabling same-document graph v3 raised actual candidate recall from
`0.904175` to `0.917765`, recovered 13 gold pairs across 12 queries, dropped
none, and raised complete-gold queries from 369 to 380. All 13 recoveries were
audited against the graph: each has an explicit same-document source-to-gold
citation reachable from a top-five seed. The treatment changed 114/438
candidate sets. With the unchanged epoch-3 reranker, fixed top-1 and top-2 and
the threshold/fallback-2 policy were exactly unchanged; top-3 F2 increased from
`0.518575` to `0.521322`, with two query wins and no losses. Pair count remained
21,900. Retrieval plus reranking time increased from 611.5 to 678.1 seconds,
plus 15.6 seconds to build the reproducible graph.

Fresh LightGBM diagnostics were mixed: fixed top-2 F2 increased from `0.562469`
to `0.577226`, while pooled threshold F2 and leakage-controlled nested score-gap
F2 decreased from `0.644395/0.648354` to `0.639982/0.643605`. Per the
predeclared protocol, this freshly fitted downstream learner does not replace
the fixed epoch-3 comparison for attributing the reference treatment.

Freeze the first stage as candidate pool 50 with graph-v3 same-document
reference expansion enabled, seed depth five, at most four base-rank-selected
expansions, and same-law neighbor expansion disabled. This choice preserves the
lower-cost pool, adds verified candidate coverage without harming the fixed
top-2 result, and exposes the rescued positives to the subsequent aligned
reranker training. Do not reopen pool size, graph scope, encoder, BM25, WRRF,
or query expansion during the two predeclared reranker treatments.

Before supervised retraining, run one checkpoint-only screen of
`AITeamVN/Vietnamese_Reranker` on this exact frozen candidate distribution.
Pin model revision `f536976248403314225d7fdfdbc87f0e9516a54e`; retain direct
retrieved Khoản–Điểm chunks, pair length 512, packed-stream batch 256, all
candidate/reference settings, fixed output cutoffs, and fresh query-grouped
five-fold OOF LightGBM. This is an initialization/checkpoint ablation, not a
reopening of first-stage selection. Do not run public/private inference or
reuse epoch-3 score thresholds before the development comparison is complete.

The checkpoint-only screen completed on 2026-10-28. The seed-42 split,
component rankings, fused article rankings, and 21,900 reranker chunk pairs
were byte-identical to the frozen epoch-3 pool-50/reference-on control; the
resolved configs differed only in output identity, graph file location, and
reranker checkpoint path. At fixed top 2, the Vietnamese checkpoint improved
precision/recall/F2 from `0.358447/0.607002/0.533073` to
`0.367580/0.615830/0.542546`. It won/lost/tied query recall on `43/36/359`
queries, but a 5,000-sample paired query bootstrap placed the F2 difference
95% interval at `[-0.019728, 0.039223]`; treat the gain as promising rather
than conclusive. Top-1 F2 improved from `0.514270` to `0.521022`, while top-3
F2 decreased from `0.521322` to `0.519751`.

Fresh five-fold OOF LightGBM fixed top-2 F2 improved from `0.577226` to
`0.597223`, and its development-tuned threshold/fallback-2 F2 improved from
`0.639982` to `0.646085`. The leakage-controlled nested-OOF score-gap estimate,
however, decreased from `0.643605` to `0.639257`. The checkpoint also handled
the thirteen gold article pairs introduced only by reference expansion worse:
epoch 3 placed three within top 5, including one at top 1, whereas the
Vietnamese checkpoint placed only one within top 5 and none within top 3; its
median rank was 25 versus 15. Therefore retain epoch 3 as the established
control and qualify `AITeamVN/Vietnamese_Reranker` as a promising alternative
fine-tuning initialization, especially for a matched initialization ablation.
Do not claim that it has already learned the frozen graph-expanded candidate
distribution. The attached Kaggle checkpoint was recorded as revision
`f536976248403314225d7fdfdbc87f0e9516a54e`, but the run artifact does not
contain an independently verified weight-file digest.

Before fine-tuning, run two final no-reference deployment controls with the
same pinned Vietnamese checkpoint and packed-stream batch 256: the historical
WRRF setting (`rrf_k=20`, BM25/Harrier/E5 `0.10/0.50/0.40`) and the retuned
setting (`rrf_k=10`, `0.15/0.55/0.30`). The completed reference-on Vietnamese
run is not a substitute for the latter. Keep pool 50, component depth 100,
`top_chunks`, split, fixed output cutoffs, and fresh five-fold OOF LightGBM
unchanged. This pair separates the fusion-weight effect from reference
expansion and preserves the historical stronger downstream configuration as a
deployment control; it does not reopen another WRRF grid.

The historical-weight Vietnamese no-reference control completed on 2026-10-29.
Its split, component rankings, fused first-stage predictions, and 21,900 chunk
candidates were byte-identical to the corresponding epoch-3 historical-weight
artifact. Vietnamese reranking improved fixed top-1/2/3 F2 over epoch 3 from
`0.513268/0.529209/0.513440` to `0.518732/0.543752/0.523416`; the top-2 paired
query bootstrap difference was `+0.014543` with a 95% interval
`[-0.013332, 0.043143]`. Its own threshold/fallback-2 F2 was `0.585713`.

Fresh LightGBM reached fixed top-2 F2 `0.591939`, threshold/fallback-2
`0.650012`, pooled score-gap `0.654008`, and leakage-controlled nested-OOF
score-gap `0.649468`. Relative to the completed new-weight/reference-on
Vietnamese run, raw reranker top-2 improved only `0.001206`, while nested-OOF
score-gap improved `0.010211`. The result confirms that the historical fusion
remains a serious deployment control, but it does not yet isolate weights from
reference expansion; await the new-weight/no-reference cell before changing
the fine-tuning distribution.

The retuned-weight Vietnamese no-reference control completed on 2026-10-29.
Its seed-42 development split and component rankings are byte-identical to the
historical-weight and retuned-weight/reference-on controls. Retuned WRRF
improved fused article Recall@50 from `0.922081` to `0.927789` and direct
reranker-candidate Recall@50 from `0.903033` to `0.904175`, but this small
coverage gain did not improve the downstream selector. Raw Vietnamese
reranker top-1/2/3 F2 was `0.521022/0.543752/0.520852`; top-2 was exactly equal
to the historical-weight control. Fresh LightGBM reached fixed top-2 F2
`0.589574`, pooled threshold/fallback-2 `0.643976`, pooled score-gap `0.640607`,
and leakage-controlled nested-OOF score-gap `0.637017`. The corresponding
historical-weight/no-reference nested estimate was `0.649468`, a difference of
`-0.012451` for the retuned weights.

With retuned weights held fixed, reference expansion raised direct candidate
Recall@50 from `0.904175` to `0.917765`, LightGBM fixed top-2 F2 from
`0.589574` to `0.597223`, and nested-OOF score-gap F2 from `0.637017` to
`0.639257`. This is a modest downstream gain despite a material ceiling gain;
it supports keeping reference-on as the graph research treatment, not claiming
that the current unfine-tuned reranker fully exploits rescued candidates. The
historical-weight/no-reference run remains the strongest measured deployment
control. The standard control evaluated only the all-row LightGBM input and
score-gap maximum cutoffs `{3,4,5}`; pooled tuning and every nested outer fold
selected maximum cutoff 3, so the reduced diagnostic grid does not explain its
lower result. Freeze these three configurations and do not reopen WRRF tuning
on this development split.

Artifact-integrity note: the later archive originally named
`vietnamese_reranker_new_weights_noref_results.tar.gz`, subsequently renamed
`vietnamese_reranker_old_weights_noref_full_results.tar.gz`, is a
historical-weight run. Although its treatment ID and experiment name say new
weights, its serialized effective
configuration is `rrf_k=20` with BM25/Harrier/E5 weights
`0.10/0.50/0.40`. Its first-stage predictions, reranker predictions, and
scoreboard match the historical-weight/no-reference run, not the true retuned
control. Do not count it as an independent run or use it to replace the valid
retuned artifact
`vietnamese_reranker_new_weights_noref_standard_results.tar.gz`, whose effective
configuration is `rrf_k=10` and `0.15/0.55/0.30`.

The corrected full-grid archive
`vietnamese_reranker_new_weights_noref_results.tar.gz` was verified on
2026-10-29. Its effective configuration is `rrf_k=10` with weights
`0.15/0.55/0.30`; its split, component rankings, first-stage predictions,
chunk candidates, reranker predictions, reranker summary, and gate summary are
byte-identical to the standard archive. Full LightGBM evaluation reproduced
fixed top-2 F2 `0.589574`, pooled threshold/fallback-2 `0.643976`, pooled
score-gap `0.640607`, and nested-OOF score-gap `0.637017`. Therefore the
standard run was not an approximation of the measured treatment, and the
first-stage decision is final.

The full LightGBM input-depth ablation contains a useful downstream result
that is separate from first-stage selection. Limiting LightGBM training and
prediction rows to reranker top 30 produced the best leakage-controlled
nested-OOF score-gap F2, `0.643248`, compared with top 10 `0.637678`, top 20
`0.635571`, and all rows `0.637017`. Top 30 also had the best pooled score-gap
F2 (`0.648649`) and threshold/fallback-2 F2 (`0.644742`). Fixed-top-2 F2 was
very slightly higher at input depth 20 (`0.594588`) than depth 10 (`0.594296`),
depth 30 (`0.591930`), or all rows (`0.589574`), so no single depth dominates
every policy. Every pooled and nested score-gap search selected output
`maximum_k=3`; larger output caps did not help. This ablation used the
retuned-weight/no-reference distribution, whereas reranker fine-tuning uses
retuned-weight/reference-on candidates. Therefore treat depth 30 as a
predeclared hypothesis and all rows as its control, not as an already selected
winner for the fine-tuning treatment. Do not fit LightGBM for every fine-tuned
epoch: select reranker snapshots first by the predeclared fixed-top-2 gate,
then compare depth 30 and all rows only for the warm start and objective
finalists on the exact reference-on candidates.

The historical-weight/no-reference depth ablation does not reproduce the same
nested-OOF winner. Its depth 10/20/30/all nested score-gap F2 values were
`0.644672/0.639692/0.647064/0.649468`, so all rows won that leakage-controlled
comparison, while depth 30 was slightly best only for pooled score-gap
(`0.655183` versus all-row `0.654008`). Depth 10 had the best fixed-top-2 F2
(`0.596763`), and depth 20 had the best threshold/fallback-2 F2 (`0.650349`).
Thus input depth is a selector hyperparameter whose effect depends on the
candidate distribution, not a universal top-30 rule. The historical treatment
also exceeded the retuned treatment at every tested nested depth, but because
it is no-reference this does not answer how historical weights would interact
with graph expansion. Do not introduce an unplanned historical-weight/ref-on
run before the frozen reranker-objective comparison.

The main reranker fine-tuning treatment is now the pointwise-BCE notebook with
pool 50, graph-v3 same-document reference expansion on, WRRF `k=10`, and
BM25/Harrier/E5 weights `0.15/0.55/0.30`. It warm-starts from pinned
`AITeamVN/Vietnamese_Reranker` weights and retains all five new adaptation-epoch
snapshots. The notebook fails closed on the verified graph digest and on the
fully resolved training configuration so a mislabeled experiment cannot run
historical weights silently. Compare its five checkpoints with fixed top-2 as
the primary checkpoint gate and fixed top-1/3 as diagnostics.

The pointwise pool-100 notebook is the active cost/ceiling ablation. It changes
only the candidate budget and artifact identities relative to pointwise pool
50: reference graph, WRRF, warm start, teacher top-30 mining
depth, five-negative exclusion, ten semi-hard negatives, optimization, and
evaluation protocol remain fixed. Pool 100/reference-on has measured candidate
Recall `0.941433`, but its matched Vietnamese warm-start result is intentionally
measured inside each notebook rather than borrowed from the pool-50 baseline.
These runs test whether matched training can exploit the larger ceiling; they
do not replace pool 50 as the predeclared main treatment unless their fixed
top-2 and subsequent nested-OOF finalist comparisons justify the doubled
reranking cost.

Measured update, 2026-10-30: pool 100 did not justify the doubled candidate
budget. The matched fixed-top-2 checkpoint screen selected pool-50 epoch 1 /
checkpoint `1208` at F2 `0.579928`; pool 100's best checkpoint reached
`0.565259`. The subsequent controlled LightGBM feature screen selected the full
61-feature `enhanced_v2` schema as the primary learned selector by nested-OOF
score-gap F2 `0.658333`. The full 69-feature `enhanced_v3` schema remains a
fixed-top-2 control at F2 `0.597850`. These results supersede the older
epoch-3/`0.10/0.50/0.40` architecture checkpoint only for the current
pointwise-finalist branch; that historical pipeline remains a frozen baseline,
not a silently rewritten configuration. The next allowed feature-pruning step
is one predeclared confirmatory reduced-schema comparison, not an open-ended
search over individual fields.

Measured update, 2026-11-01: the predeclared fold-safe learned stopping policy
on the frozen pool-50 checkpoint-1208, `enhanced_v2` ranking scored `0.6672` on
the Codabench public split (submission `1025830`) and `0.6123` on private
(submission `1025816`). This exceeds the frozen score-gap control by
`0.0121/0.0076` and threshold/fallback 2 by `0.0148/0.0069` on public/private.
The consistent direction across both hidden splits supports the hypothesis that
query-conditional prefix stopping improves cardinality selection. It does not
authorize leaderboard tuning: select it for the final report only after copying
and checking its saved nested-OOF development result and provenance from the
stopping artifact.

One and only one follow-up cardinality treatment is predeclared: a fold-safe
listwise prefix-utility Ridge model with `alpha=10`, maximum `k=5`, per-query
prefix-F2 regression targets, and smaller-prefix tie breaking. It reuses the
identical `enhanced_v2` OOF rankings and five outer folds as learned stopping.
Its purpose is to test whether simultaneous comparison of all prefixes avoids
error propagation from sequential continue/stop decisions. Do not open a
hyperparameter search or submit it based only on leaderboard expectations;
nested-OOF must determine whether it merits external evaluation.

Measured update, 2026-11-01: the frozen listwise treatment scored `0.6661`
public (submission `1025919`) and `0.6245` private (submission `1025907`). It is
`0.0011` below learned stopping on public and `0.0122` above it on private; its
public-private gap is `0.0416` versus stopping's `0.0549`. This supports further
offline analysis of prefix selection, but does not authorize choosing Ridge
alpha, feature sets, or additional policies from leaderboard outcomes. Read the
two saved nested-OOF artifacts before selecting the report finalist.

The final allowed cardinality expansion is a closed five-family nested screen,
not an open-ended model zoo. Its complete grids are frozen in
`configs/cardinality_policy_screen.yaml`; every family performs inner selection
inside each outer query fold and keeps ranking, seed, split, and maximum `k=5`
fixed. Although the runner packages every case for reproducibility, only
`selected_by_nested_oof.json` can select a report finalist. Any comparison of
all case ZIPs on Codabench must be labeled exploratory repeated evaluation.

Measured update, 2026-11-01: the closed screen selected balanced multiclass
logistic regression (`C=0.1`) by nested-OOF F2 `0.669557`, only `0.000887` above
tuned Ridge utility (`alpha=1`, F2 `0.668670`). The winner's Precision/Recall
were `0.556355/0.705441`, mean output size was `2.315`, and its exact/under/over
oracle-k rates were `0.484018/0.130137/0.385845`. The nonlinear multiclass model
achieved higher exact-k accuracy (`0.625571`) but only F2 `0.605612` because it
systematically chose short prefixes. The scientific winner is therefore
multiclass logistic, while the tiny margin over Ridge must be reported rather
than described as a decisive algorithmic advantage.

Official confirmation, 2026-11-01: tuned Ridge utility scored `0.6700/0.6265`
on public/private (submissions `1025993/1025992`), while the nested-selected
balanced multiclass logistic policy scored `0.6847/0.6368` (submissions
`1025984/1025979`). Thus the development-selected ordering transferred to both
hidden splits, with multiclass gains of `0.0147/0.0103` over Ridge. The three
remaining screen families are closed as development-only negative results and
must not be submitted merely to search the leaderboard.

One focused follow-up to the selected multiclass policy is frozen in
`configs/cardinality_confirmatory.yaml`: unchanged balanced multiclass,
cost-sensitive expected-F2 decoding, and an inner-tuned multiclass–Ridge
ensemble with mixing weights `{0.25,0.5,0.75}`. No additional model family or
grid is allowed in this confirmatory notebook. It persists every fold record,
deployment decision, coefficient, utility matrix, prediction, hash, and ZIP;
future reporting should read the artifact rather than rerun it.

Measured confirmatory result, 2026-11-01: multiclass–Ridge (`lambda=0.75`)
reached nested-OOF F2 `0.670230` versus fixed multiclass `0.669890`, then scored
`0.6863/0.6354` public/private (submissions `1026035/1026030`) versus
`0.6847/0.6368` for plain multiclass. The directions split and all differences
are at most `0.0016`; retain the simpler multiclass policy. Cost-sensitive
decoding collapsed to F2 `0.510219` because mean output size rose to `4.657534`.
These results close cardinality-model development; do not tune further from the
leaderboard.

User-authorized final diagnostic, 2026-11-02: cardinality model development
remains closed, but one controlled change to the *shape* of the returned set is
allowed. `configs/final_evidence_subset.yaml` tests exactly two non-prefix
selectors over the already frozen top-five LightGBM ranking: multiclass
cardinality plus binary candidate selection, and joint Ridge utility over all
31 nonempty subsets. It also measures oracle prefix versus oracle subset. No
retrieval, reranking, feature, split, or maximum-output setting changes, and
selection remains nested-OOF only. After this run, freeze the decision layer;
do not add families or tune from public/private outcomes.

Measured hidden-split update, 2026-11-02: cardinality plus binary subset
selection scored `0.6847/0.6368` on public/private (submissions
`1026117/1026110`), exactly matching plain multiclass to four reported decimal
places. Joint subset utility scored `0.6685/0.6337` (`1026118/1026111`), lower
than plain multiclass by `0.0162/0.0031`. Equal aggregate scores do not prove
query-level predictions are identical; verify the two `results.json` files
before making that stronger claim. No non-prefix method earns selection from
these hidden-split results, and plain multiclass remains the parsimonious final
policy subject to the predeclared nested-OOF record.

Measured representation update, 2026-11-02: the matched fixed-window 448/64,
reference-off pipeline scored `0.6669/0.6520` with threshold selection
(`1026162/1026154`) and `0.6739/0.6580` with score gap
(`1026161/1026153`) on public/private. Its score-gap variant substantially
improves over the hierarchical reference-on score-gap result, especially on
private, but comparison with the final hierarchical multiclass result is
confounded by decision policy. If one final experiment is run, it must apply
the already frozen balanced multiclass protocol to the fixed-window OOF and
test rankings; do not tune that choice from the private improvement.

The policy-matched run completed on 2026-11-02. Fixed-window balanced
multiclass scored `0.6833/0.6677` public/private (`1026181/1026178`), compared
with hierarchical multiclass `0.6847/0.6368`. Fixed-window multiclass–Ridge
scored `0.6933/0.6685` (`1026176/1026175`), the then-highest historical-WRRF
project pair.
Because the ensemble's gain over fixed multiclass is asymmetric
(`+0.0100/+0.0008`), select it only if the fixed run's predeclared nested-OOF
comparison also ranks it first. Regardless of policy, the matched evidence now
supports fixed 448/64, reference-off as the more robust representation.

R3-retuned fixed-window update, 2026-11-02: five-fold cross-fitted first-stage
selection chose the full-development WRRF deployment ratio BM25/Harrier/E5
`0.15/0.65/0.20`. Refitting the complete fixed/reference-off pipeline and the
same confirmatory policy family produced multiclass–Ridge nested-OOF
Precision/Recall/F2 `0.5449/0.7280/0.6822`, mean output size `2.3584`, and
deployment `lambda_multiclass=0.5`. Public/private Macro-F2 reached
`0.7055/0.6735` (submissions `1026275/1026274`), improving over the historical
WRRF Ridge run by `+0.0122/+0.0050`. The fusion ratio was selected without
public/private labels; hidden scores are reporting-only confirmation.

Interpretation boundary: epoch-1 checkpoint 1208 was trained with article-level
supervision on hierarchical Khoản–Điểm chunk bags, not fixed-window bags. The
fixed pipeline changes the chunks scored at inference while deliberately
holding the neural checkpoint constant. This supports fixed-window robustness
and cross-representation transfer, but a claim about fixed-window supervised
training requires a separate retraining ablation.

That ablation is implemented in
`notebooks/training/kaggle_finetune_reranker_fixed_refoff_3epochs_sched5.ipynb`.
It keeps pool 50, seed 42, WRRF10, the false-negative-aware article-bag mining
policy, pointwise MIL BCE, four cycled passages per bag, pair length 512,
effective batch 16, and learning rate `2e-5`. It uses fixed token windows
448/64, references off, and cross-fitted R3 BM25/Harrier/E5 weights
`0.15/0.65/0.20`, then trains three adaptation epochs on the five-epoch
scheduler horizon. The checkpoint-1234 public/private submissions are now
measured below; archive ingestion and a saved checkpoint-1234 development
policy comparison remain outstanding.

Owner-reported official submissions for the new fixed-window-trained epoch-1
checkpoint `1234`, received 2026-11-02, scored balanced multiclass
`0.6967/0.6809` public/private (IDs `1027658/1027657`) and multiclass–Ridge
`0.6978/0.6825` (IDs `1027650/1027647`). The project now uses the Ridge pair as
its practical baseline because private performance is prioritized; this is a
post-hoc operational designation based on hidden leaderboard results, not an
unbiased method-selection conclusion. The earlier checkpoint-1208 run remains
the public-score record at `0.7055`, while checkpoint 1234 has the highest
reported private score at `0.6825`. Preserve both results in comparisons and do
not continue tuning against private feedback. Archive ingestion and a saved
checkpoint-1234 development policy comparison remain necessary to support the
methodological account.
