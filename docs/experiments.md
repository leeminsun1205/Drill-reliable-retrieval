# Experiments and artifacts

## Pipeline boundaries

Keep the pipeline separable:

```text
data preparation -> retrieval -> fusion -> reranking -> evaluation -> analysis
```

Retrievers return ranked article IDs and scores; they do not evaluate their own
output. Reranking consumes a fixed first-stage candidate pool and remains a
separate stage.

Dense retrieval must encode corpus and queries independently, retain the mapping
from vector positions to `aid`, support batching, and avoid unnecessary corpus
re-encoding.

For Sentence Transformers multi-GPU encoding, the parent process loads the
source model on CPU before spawning one worker per configured CUDA device. This
prevents the parent model from occupying GPU 0 alongside the GPU-0 worker. A
Kaggle T4x2 process trace on 2026-10-27 verified the previous failure mode: the
parent `run_experiment.py` process held 13,576 MiB on GPU 0 while its GPU-0
worker attempted to materialize another model replica and ran out of memory;
the GPU-1 worker completed its chunks. This is a runtime lifecycle correction,
not a retrieval-method or scoring change.

The active exact-search implementation uses FAISS `IndexFlatIP` over float32
vectors, with explicit L2 normalization for cosine similarity. This remains
exhaustive exact KNN, not ANN. `exact_numpy` is retained as the reference
backend. FAISS search adaptively expands chunk depth until article-level max
aggregation has enough unique aids and every score tie at the selected boundary
is covered, then applies the same deterministic ascending-`aid` tie-break.
Parity tests require identical chunk/article IDs and float32 scores within
`rtol=1e-5`, `atol=1e-6`. A local 50,000-vector, 128-dimensional, 64-query
diagnostic measured 0.089 seconds for FAISS CPU versus 0.181 seconds for the
NumPy reference; this is a synthetic implementation diagnostic, not a DRiLL
runtime result. Active configs use FAISS CPU so vector search does not compete
with dense encoders or the reranker for GPU memory.

Fusion must be explicit. Reciprocal Rank Fusion is suitable for rank-based
fusion. Direct weighted fusion of BM25 scores and cosine similarities requires a
documented normalization strategy.

The repository's weighted RRF uses one-based ranks and contributes
`weight / (rrf_k + rank)` once per unique `aid` in each component. This matches
the standard weighted-RRF arithmetic used by LangChain's ensemble retriever.
The repository intentionally differs in surrounding policy: identity is always
the legal article `aid` (or exact `(aid, chunk text)` for multi-dense chunk
fusion), sparse zero-score fillers are excluded by default, component depths are
explicit, and final score ties use ascending `aid`. These details can change a
ranking even when two systems cite the same RRF formula. Do not compare RRF
implementations without fixing `rrf_k`, one- versus zero-based rank, weights,
deduplication key, per-component depth, missing-document behavior, zero-score
handling, and tie-breaking.

Sparse zero-score exclusion is a pipeline policy, not a theorem or mandatory
part of RRF. The native retriever pads with deterministic zero-score articles
only when fewer than the requested `top_k` documents receive a positive BM25
score; treating those placeholders as retrieved lexical matches would give
arbitrary articles an RRF contribution. Standard Lucene-formula BM25 in this
project has positive IDF and term-frequency contributions, so an exact zero
means no retained query-term contribution. The policy is configurable through
`include_zero_score_sparse`. In the measured 438-query seed-42 development run,
both native and bm25s had 100 positive scores in every saved top-100 ranking,
so zero-score exclusion affected zero development queries and cannot explain
any metric difference in that comparison. Re-audit this fact for other query
sets or scoring variants rather than assuming it universally.

Standard BM25 uses the `bm25s` backend after the replacement gate reported
below passed. It receives the
same project-generated VnCoreNLP tokens and vocabulary-frequency pruning, uses
`method: lucene` plus Lucene IDF, and retains deterministic ascending-`aid`
ties. The native backend remains the correctness oracle, and
`notebooks/archive/kaggle_bm25_native_vs_bm25s.ipynb` reproduces the complete parity
check on Kaggle. Raw scores are not a parity criterion because bm25s may omit a
positive rank-invariant scale factor; rank-based fusion is unaffected by that
factor.
BM25+ is outside this replacement gate because native and library conventions
for the non-occurrence delta term differ.

The standard-BM25 replacement gate was executed locally on 2026-10-27 with the
59,628-article local corpus, all 438 seed-42 development queries, VnCoreNLP,
`k1=1.2`, `b=0.75`, and candidate depth 100. Native and bm25s produced
identical ordered rankings for 438/438 queries at every checked cutoff
1/2/3/5/10/20/50/100; therefore all 43,800 positions in the saved top-100
rankings matched. Precision, recall, F2, and Recall@1/5/10/20/50/100 were
identical, including F2@3 `0.3799077303` and Recall@100 `0.8774516199`.

On the same local Linux/Python 3.12.9 environment, uncached end-to-end runtime
was 341.36 seconds for native and 295.43 seconds for bm25s; this measurement is
dominated by repeated VnCoreNLP corpus segmentation. With verified cache hits,
runtime was 37.49 versus 9.16 seconds, a 4.09x end-to-end speedup for the sparse
evaluation run. Cache size was 73.78 MiB versus 55.26 MiB, a 25.1% reduction.
These are measured local implementation results, not Kaggle runtime claims.
The evidence qualifies bm25s to become the execution backend for standard BM25;
retain native as a reproducibility oracle and for currently native-only RM3 and
BM25+ semantics until those treatments receive separate parity definitions.

For controlled multi-dense experiments, the factory accepts a base
`retrieval.dense` mapping, named partial overrides under
`retrieval.dense_profiles`, and a non-empty
`retrieval.hybrid.dense_components` list. Each selected component declares a
unique `name`, `profile`, `weight`, and optional `top_k`. WRRF combines the
sparse rank and every dense rank; saved component names are `sparse` and
`dense:<name>`. Exact duplicate chunks, identified by parent `aid` and text, add
their dense rank contributions before direct-chunk reranking. Non-identical
chunks remain distinct evidence. Omitting `dense_components` preserves the
original one-sparse/one-dense behavior and artifact schema.

Retrieval cache preparation is a separate, non-evaluating stage. Run
`scripts/prepare_retrieval_cache.py` once per component so each dense model exits
before the next is loaded. Supported component names are `sparse`, `dense` for
a single-dense config, or names under `retrieval.hybrid.dense_components`.
Dense caches retain the validated embedding matrix and ordered `aid` mapping.
Sparse caches store the fitted BM25 index as JSON metadata/terms plus NumPy
integer arrays; pickle is not used. Both cache types validate a deterministic
corpus fingerprint and a configuration key before reuse.

Set `cache.require_hit: true` in evaluation and inference configs after cache
preparation. A missing or incompatible cache then fails immediately instead of
silently rebuilding inside a long experiment. Cache preparation produces no
metric and is not itself an experiment result.

Example peer selection after independent screening:

```yaml
retrieval:
  hybrid:
    fusion: weighted_rrf
    rrf_k: 20
    sparse_weight: 0.1
    sparse_top_k: 100
    dense_components:
      - {name: vietlegal_harrier, profile: vietlegal_harrier, weight: 0.5, top_k: 100}
      - {name: multilingual_e5_large, profile: multilingual_e5_large, weight: 0.4, top_k: 100}
  dense:
    chunking:
      method: khoan_diem
      max_tokens: 448
      overlap_tokens: 64
      tokenizer_model_name: BAAI/bge-m3
      tokenizer_document_prefix: ""
    encoder:
      backend: sentence_transformers
      max_seq_length: 512
  dense_profiles:
    vietlegal_harrier:
      encoder:
        model_name: mainguyen9/vietlegal-harrier-0.6b
        query_prefix: "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "
        document_prefix: ""
        trust_remote_code: true
      cache: {directory: embeddings/vietlegal_harrier_0_6b}
    multilingual_e5_large:
      encoder:
        model_name: intfloat/multilingual-e5-large
        query_prefix: "query: "
        document_prefix: "passage: "
      cache: {directory: embeddings/multilingual_e5_large}
```

The peer list is structurally unbounded, although the active controlled run uses
exactly the two models shown. The owner-selected primary WRRF ratio is
BM25/Harrier/E5-large `1:5:4`, stored as normalized weights `0.1/0.5/0.4`;
normalization changes only the common score scale, not ranking. This is a
planned setting, not a measured winner. Model-specific query
instructions and pooling must follow the verified model card rather than being
silently guessed. The historical thirteen-feature LightGBM path expects one
component named `dense`. Multi-dense runs extend that base schema with a
raw-score and reciprocal-rank pair for each stable `dense:<name>` component,
ordered by component name; a two-dense run therefore had 15 base features. The
The `base_v1` schema exposes those 15 historical features directly for a
two-dense run. The historical enhanced-v1 schema appends 26 deterministic
features covering within-query
normalization and adjacent score gaps, retriever agreement, explicit-reference
provenance, chunk fusion statistics, and query legal-marker counts. A two-dense
run therefore has 41 features. Enhanced-v2 adds 20 features derived from the
actual scored reranker chunks: score distribution and margin, best-chunk input
position, chunk lengths, query/chunk lexical coverage, and legal-citation
overlap. Enhanced-v3 adds eight corpus-static article features: character and
token counts, vocabulary size, structural marker counts, legal-citation count,
and digit-token fraction. The corresponding two-dense schemas contain 61 and
69 features. Missing optional retrieval feedback maps only
the reference-provenance fields to zero and never changes feature order. Dev
and test component schemas must match exactly, and every multi-dense treatment
requires fresh query-grouped OOF training rather than reuse of the baseline
LightGBM model.

The active peer pair selected by the project owner on 2026-10-26 is
`mainguyen9/vietlegal-harrier-0.6b` plus
`intfloat/multilingual-e5-large`. Their query/document instructions match the
completed standalone screen. The controlled config retains BGE-M3-tokenized
Khoản–Điểm 448/64 chunk boundaries for both encoders so this treatment changes
embeddings rather than segmentation; the tokenizer fixes representation
boundaries and does not make BGE-M3 an ensemble anchor.

The multi-dense notebook carries this pair through the strongest existing
epoch-3 reranker, freshly trained multi-component LightGBM, and score-gap final
selection; it does not modify or reuse the single-dense LightGBM notebook.
Standard BM25 is the first sparse control. BM25+ with delta `1.0` is the next
isolated treatment in the same notebook, after the control completes, because
changing the dense set and sparse scoring rule in one run would not identify
which change caused the result.

On 2026-10-27, the first selected multi-dense reranker run completed first-stage
retrieval and the WRRF grid but then remained at 81% of RoBERTa checkpoint
materialization for roughly ten hours. The selected `0.1/0.5/0.4`
BM25/Harrier/E5-large case had already measured Recall@50 `0.922081`,
Recall@100 `0.951653`, fixed top-2 F2 `0.518868`, and fixed top-3 F2 `0.514429`.
No reranker metric was produced. The last logs showed both 205,898-by-1,024
dense indexes and encoder state still resident immediately before loading the
epoch-3 reranker, so the event is recorded as a probable resource-lifecycle
stall, not a model-quality result or a proven CUDA OOM.

The experiment and submission runners now snapshot retrieval provenance and
component rankings, explicitly release every dense encoder, embedding matrix,
query cache, and chunk list, run Python garbage collection and CUDA cache
cleanup, and only then load the cross-encoder. The fix preserves retrieved
chunks and rankings and therefore does not change the method. A repaired run
must emit `Releasing retrieval resources before reranker load` before checkpoint
materialization; completion on Kaggle remains pending.

The same-process handoff additionally reports per-device free, allocated, and
reserved CUDA memory immediately before and after retrieval release. It drops
the top-level retriever reference, synchronizes both devices, reclaims CUDA IPC
allocations left by SentenceTransformers workers, and empties the allocator
cache before constructing the reranker. Reranker logs distinguish CPU checkpoint
materialization, transfer to the primary CUDA device, and readiness for
inference. These are resource-lifecycle and diagnostic changes only; retrieval
rankings, reranker inputs, model identifiers, and scoring semantics are
unchanged. The operational reranker mini-batch is explicitly configured at 64
for the current T4x2 verification run.

The 2026-10-27 lifecycle audit found an integration error in that correction:
submission inference used the strong barrier, but the failing selected
development command invokes `scripts/run_experiment.py`, whose experiment
runner still used only each retriever's local `close()` method. The experiment
and submission runners now call one shared barrier implementation. A separate
`cuda_lifecycle_trace.jsonl` records global NVML process memory plus this
process's PyTorch allocated/reserved memory at process start, retriever build,
each sparse/dense fit, each dense cache load, component query retrieval, worker
completion, candidate completion, before/after retrieval close, before reranker
construction, and after reranking. The multi-dense notebook packages this trace.
The subsequent T4x2 trace showed that the barrier did run but NVML stayed at
11,300 MiB on GPU 0 before and after it. The allocation belonged to the
experiment PID while that PID's PyTorch allocator reported zero allocated and
reserved bytes; shutdown also reported four leaked multiprocessing semaphores.
The repeated three checkpoint loads match a SentenceTransformers source model
plus its two multi-device encoding workers. This localizes the remaining leak
to native CUDA state from the internal multi-device encoding lifecycle rather
than live tensors reachable by the experiment runner. Multi-GPU dense query
encoding now runs below a disposable owner process that is joined before
retrieval continues. Process exit provides the CUDA resource boundary while
the returned NumPy embeddings and all ranking/scoring logic remain unchanged.
The next T4x2 run must still measure the expected VRAM drop before this runtime
fix is considered hardware-validated.

That validation still showed 11,300 MiB attached to the retrieval experiment
PID after the isolated encoder returned. The selected-dev runner therefore now
uses a hard retrieval-to-reranking process-image boundary: it atomically writes
`retrieval_stage_handoff.json`, containing first-stage ranks and scores, complete
chunk records and mappings, component ranks, feedback records, and retriever
metadata, then replaces itself with a fresh Python image via `exec`. The resumed
image validates the handoff schema and performs only reranking and downstream
evaluation. Unlike allocator cleanup, process-image replacement closes every
CUDA driver resource owned by retrieval while preserving the original command,
exit status, artifact directory, candidates, and ranking semantics.

On 2026-10-26, the project owner reported an isolated Kaggle T4 smoke test over
the same first 128 articles, which produced 505 Khoản–Điểm representations for
each encoder. BGE-M3 completed in 85.27 seconds on one GPU and 54.79 seconds on
two GPUs; VietLegal-E5 completed in 75.82 seconds on one GPU and 54.45 seconds
on two GPUs. Every output had shape `[505, 1024]`. These are measured diagnostic
timings, not retrieval-quality results. They show that both checkpoints and the
SentenceTransformers two-GPU path work at small scale; they do not rule out a
large-input queue/memory problem or interference when the two encoders run
sequentially in one process. A matched sequential two-GPU test also completed:
BGE-M3 took 53.28 seconds, VietLegal-E5 then took 42.10 seconds, and the total
was 96.63 seconds. Parent-process peak RSS rose from 5.20 GiB after BGE-M3 to
7.95 GiB after VietLegal-E5; this measurement excludes the worker processes.
Python's resource tracker reported four leaked semaphore objects at process
shutdown. The warning is diagnostic evidence of incompletely reclaimed
multiprocessing resources, not proof that it caused the full-corpus timeout.
Larger samples and total process-tree memory measurement remain pending.

The active Kaggle multi-dense notebook now enables a diagnostic-only CUDA
process trace at `multi_dense_selection/cuda_process_trace.jsonl`. From before
each SentenceTransformer model load until encode returns or raises, it samples
`nvidia-smi` every 0.5 seconds and records GPU UUID, PID, PPID, command, and
used device memory. This does not change device placement, batching, retrieval,
or scoring. Its purpose is to distinguish parent-model residency, duplicate
worker placement, and batch-time growth before any lifecycle fix is selected.

#### Standalone dense-screening results

On 2026-10-26, the project owner reported completion of all seven standalone
dense runs on the fixed 438-query seed-42 development split. The runs were
reported as using the screening protocol documented in `docs/research_plan.md`: BGE-M3-tokenized
Khoản–Điểm 448/64 representations, exact cosine search, candidate depth 100,
and no BM25, fusion, or reranker. The following values are measured run outputs
reported by the owner, but remain pending inspection of the downloaded
`dense_screening_*.tar.gz` archives, including their configs, split IDs,
predictions, rankings, and provenance.

| Model | Best output k | Precision | Recall | F2 | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | 2 | 0.309361 | 0.526826 | 0.461889 | 0.411149 | 0.663775 | 0.747624 | 0.828104 | 0.878816 | 0.910421 |
| `intfloat/multilingual-e5-large` | 2 | 0.311644 | 0.529322 | 0.464441 | 0.402664 | 0.673157 | 0.761704 | 0.828696 | 0.896015 | 0.929447 |
| `nhminh107/VietRAG-Embed` | 2 | 0.286530 | 0.489346 | 0.428661 | 0.350647 | 0.616955 | 0.699717 | 0.755730 | 0.848543 | 0.883496 |
| `mainguyen9/vietlegal-e5` | 3 | 0.242770 | 0.610997 | 0.468788 | 0.400571 | 0.696804 | 0.781279 | 0.850500 | 0.919145 | **0.946760** |
| `mainguyen9/vietlegal-harrier-0.6b` | 2 | 0.337900 | 0.581050 | **0.507947** | **0.456240** | **0.704338** | **0.803050** | **0.866862** | **0.924364** | 0.946325 |
| `bqbbao6/vietnamese-legal-embedding` | 2 | 0.311644 | 0.519597 | 0.458418 | 0.409056 | 0.675875 | 0.762139 | 0.844711 | 0.897380 | 0.928800 |
| `truro7/vn-law-embedding` | 2 | 0.287671 | 0.483828 | 0.425764 | 0.356164 | 0.624049 | 0.703006 | 0.790030 | 0.859252 | 0.909426 |

For completeness, the reported macro-F2 sweep at every fixed output size is:

| Model | k=1 | k=2 | k=3 | k=4 | k=5 | k=10 |
|---|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | 0.422127 | **0.461889** | 0.461228 | 0.443455 | 0.409414 | 0.310329 |
| `intfloat/multilingual-e5-large` | 0.412781 | **0.464441** | 0.462161 | 0.438793 | 0.412249 | 0.314301 |
| `nhminh107/VietRAG-Embed` | 0.360177 | **0.428661** | 0.424328 | 0.398340 | 0.377566 | 0.286898 |
| `mainguyen9/vietlegal-e5` | 0.412815 | 0.468025 | **0.468788** | 0.448016 | 0.429426 | 0.325152 |
| `mainguyen9/vietlegal-harrier-0.6b` | 0.468166 | **0.507947** | 0.481232 | 0.451163 | 0.431702 | 0.332737 |
| `bqbbao6/vietnamese-legal-embedding` | 0.421078 | **0.458418** | 0.451936 | 0.434285 | 0.415435 | 0.316912 |
| `truro7/vn-law-embedding` | 0.366283 | **0.425764** | 0.421291 | 0.411025 | 0.385605 | 0.291446 |

The best-output row is selected independently for each model from fixed
`output_k` values 1, 2, 3, 4, 5, and 10. VietLegal-Harrier has the highest
reported best F2 and candidate recall at every recorded cutoff through 50.
VietLegal-E5 is ahead at Recall@100 by only `0.000435` and is second on best F2.
These aggregate results make both legal-domain models plausible follow-up
candidates, but do not establish error complementarity. Selection for a
multi-dense treatment still requires the planned per-query overlap, uniquely
recovered gold, and oracle-union analysis against BGE-M3. Runtime, peak memory,
resolved revisions, embedding dimensions, prefix/pooling details, and remote-
code status were not included in this report and must be recovered from the
archives or recorded separately before the screen is considered fully verified.

The dense baseline is decomposed into encoder, cache, and search components in
`configs/dense.yaml`. It uses `intfloat/multilingual-e5-base`, explicit E5 query
and passage prefixes, a recorded maximum sequence length, and exact cosine
search. The FAISS Flat implementation is checked against the NumPy exact
reference before any future approximate backend is considered. Corpus
embeddings are cached only after validating the corpus fingerprint, ordered aid
mapping, and document-encoder configuration.

Model trials reuse `configs/dense.yaml` and override experiment leaves through
CLI flags. Common dense fields have readable flags such as `--model-name`, while
repeatable `--set PATH=VALUE` supports less common fields without creating a
YAML file per trial. Unknown paths are rejected and the fully resolved config is
still saved in the run artifact.

The current dense baseline has no chunking stage: one embedding is created per
article and the tokenizer truncates at `max_seq_length`. This limitation must be
kept distinct from a later chunking treatment, which will need explicit chunk
boundaries, overlap, chunk-to-article mapping, and score aggregation.

The runner now represents that control explicitly as
`retrieval.dense.chunking.method: article_truncate`. The first implemented
treatment is `token_window`: it uses the exact fast tokenizer identified by the
dense encoder config, subtracts document-prefix and model special tokens from
the total `max_tokens` budget, creates overlapping windows with raw character
offsets, and retains `chunk_id -> aid`. Exact search aggregates all chunk scores
to unique article IDs with `max` before applying article-level top-k. No
structure-aware boundary claim is made for these fixed windows.

The prefix/special-token subtraction is only the initial window estimate. Every
candidate chunk is then tokenized again as the exact full encoder input
`document_prefix + chunk_text` with special tokens enabled. If it exceeds
`max_tokens`, the window is shortened until the invariant holds. Run metadata
records `max_encoded_tokens_observed`; a 512-token treatment must therefore
report a value no greater than 512.

The implemented `recursive_token` baseline keeps the same tokenizer, total
budget, overlap, aggregation, and exact full-input post-check as `token_window`.
For each candidate window it recursively falls back through paragraph, newline,
sentence-ending punctuation, whitespace, and finally a hard token boundary. A
`minimum_fill_ratio` (0.5 by default) prevents an early separator from producing
an unnecessarily small chunk. This is a deterministic in-repository recursive-
separator baseline, not a claim of byte-for-byte LangChain equivalence. Run
metadata records how many final boundaries came from each separator level.

The implemented `khoan` treatment uses `conservative_line_sequence_v1`. Parsed
Khoản are contiguous raw spans; an unnumbered article lead-in is retained in the
first span. Oversized Khoản use the same tokenizer-window fallback and overlap
as the fixed baseline. Articles with no accepted structure or ambiguous
numbering fall back to whole-article token windows, so text is never discarded
or assigned fabricated boundaries. Parse-status counts, oversized fallback
windows, maximum encoded length, and maximum chunks per article are recorded in
run metadata.

The implemented `diem` treatment first applies the same conservative Khoản
parser, then accepts only consecutive Vietnamese lettered Điểm sequences within
each accepted Khoản. Each Điểm representation repeats the raw Khoản lead-in and
records the lead-in and Điểm as separate source spans. Long Điểm are split with
the exact model tokenizer while repeating the lead-in in every window. A Khoản
with no accepted Điểm, ambiguous point numbering, or a lead-in that exhausts
the token budget falls back to raw Khoản token windows; an article whose Khoản
structure is unavailable falls back to article token windows. Thus the
treatment changes only representation granularity and preserves the same
512/64 budget, max chunk-to-aid aggregation, and lossless fallback policy used
by the Khoản experiment.

The chunk cache key includes the model/revision, tokenizer-window budget,
overlap, document prefix, and chunker version. Run metadata records the content
token budget, chunk count, maximum chunks per article, representation count,
unique aid count, and aggregation method. Changing the representation therefore
cannot silently reuse article-level embeddings.

The implemented shared dense chunk cache is additionally keyed by the ordered
raw-corpus fingerprint and stores every `Chunk` field in checksummed UTF-8
JSONL. Harrier and multilingual-E5 share this artifact because their controlled
multi-dense treatment freezes boundaries to the same BGE-M3 Khoản–Điểm 448/64
chunker. Existing embedding caches remain compatible: on a chunk-cache miss the
first dense component deterministically rebuilds and saves the table, while the
second component and later processes load it. Embedding matrices remain
model-specific and continue to validate the reconstructed chunk fingerprint and
ordered parent-`aid` mapping.

Length settings are tokenizer-token counts, not words or characters. A dense
bi-encoder encodes query and document independently, so each side has its own
maximum. A cross-encoder instead encodes a query-document pair; its configured
`max_length` is the combined budget for query tokens, document tokens, and
model-specific special tokens. Reranking must therefore preserve the query and
truncate only the second sequence when truncation is unavoidable. Production
chunk preparation must use the exact model tokenizer and record its model ID,
revision, token budget, and overlap. Whitespace word counts remain diagnostic
proxies only.

The initial controlled chunking matrix is planned as: article truncation,
fixed-window, Khoan-level with tokenizer-window fallback, and Diem plus Khoan
lead-in with the same fallback. A 512-token document window with 64-token
overlap is the cost-conscious starting point; BGE-M3 1,024-token windows with
128-token overlap are an ablation rather than a universal default. Before that
ablation, BGE-M3 must be run with the winning Điểm representation at 512/64 so
the first E5-versus-BGE comparison changes only the encoder. For
`BAAI/bge-reranker-v2-m3`, 1,024 pair tokens are the initial operational setting
because that is the length reported for fine-tuning and recommended by its
authors, despite the architecture accepting up to 8,192. The exact query cap
must be selected from measured training-query token lengths before the
reranker experiment; 192 tokens is only a provisional engineering value, not a
settled dataset fact.

The measured multilingual-E5-base audit on 2026-09-29 supports retaining the
512-token article-truncation run as a control but not as the only dense
baseline: 17.03% of non-blank articles exceed its total input budget. Queries
are short (maximum 58 total tokens with the E5 query prefix and special tokens),
so document truncation is the immediate concern for this bi-encoder. The fixed
512/64 token-window experiment should therefore precede recursive or legal-
structure-aware treatments. Do not infer an exact chunk count from length
percentiles; record it from the executed chunker metadata.

The implemented first hybrid method is weighted reciprocal rank fusion (WRRF;
also sometimes shortened to weighted RRF), not direct addition of raw BM25 and
cosine scores. For one-indexed component ranks, the fused score is
`w_sparse/(rrf_k + rank_sparse) + w_dense/(rrf_k + rank_dense)`, omitting a
term when the article is absent from that component. Ties are resolved by
ascending `aid`. Sparse and dense candidate depths, weights, and `rrf_k` are
explicit configuration values and must be tuned only on the development split.

BM25's standalone implementation fills a fixed-depth list with score-zero aids
when too few articles have lexical overlap. Hybrid fusion excludes those
arbitrary sparse fillers by default through `include_zero_score_sparse: false`;
the original component ranking is still written to
`component_rankings.jsonl`. The hybrid run metadata nests both retrievers'
metadata, so model, chunking, BM25 preprocessing, cache behavior, and fusion
parameters remain inspectable. `configs/hybrid.yaml` provides the equal-weight
`rrf_k=60` starting point; it is not a claim that those values are optimal.

BM25 preprocessing is part of the method configuration. The initial ablation
compared the NFKC/lowercase regex baseline with Underthesea, PyVi, and VnCoreNLP
while holding `k1`, `b`, candidate depth, and output selection fixed. The
maintained implementation now retains regex, PyVi, and the best measured
VnCoreNLP treatment; Underthesea and character/whitespace tokenizer modes were
removed to reduce unsupported experimental branches. Dense model tokenization
and chunking remain separate experiments and must not silently inherit the BM25
word-segmented representation.

Stopword and document-frequency treatments must be evaluated after fixing the
word segmenter. The initial matrix is: no stopwords/pruning, conservative legal
stopwords, the broader legacy stopwords, `min_df` pruning, and `max_df` pruning.
Change one factor at a time and initially retain the same BM25 parameters and
output policy. Because BM25 IDF already suppresses ubiquitous terms, pruning is
an empirical treatment rather than an assumed improvement. Inspect candidate
Recall@100 as well as final F2 before selecting a treatment.

RM3 and dense Rocchio are implemented as independently disabled pseudo-
relevance-feedback treatments. RM3 uses the configured BM25 token processor,
an initial BM25 pass, a Dirichlet-smoothed relevance model over the top feedback
articles, and explicit interpolation with the original query model. It scores
the second pass with weighted query terms; it does not approximate weights by
repeating strings. Rocchio performs an initial dense chunk search, selects
feedback representations from distinct `aid` values, combines their normalized
centroid with the original query vector, and searches again. Negative feedback
is disabled by default because low-ranked first-pass candidates are not known
non-relevant examples.

The two components are orthogonal: RM3 changes only the sparse branch, Rocchio
changes only the dense branch, and enabling both changes both inputs before the
unchanged WRRF and reranker. Every feedback run writes the selected feedback
article IDs; RM3 additionally writes expanded terms and weights. Compare
`none`, `RM3`, `Rocchio`, and `RM3 + Rocchio` with identical candidate budgets,
WRRF, epoch-3 reranker, and output policy. These are generic expansion controls
for the later explicit-reference treatment, not legal-reference methods.

The first owner-run Kaggle screen on 2026-10-08 found that conservative RM3
produced the same selected epoch-3 threshold/fallback-2 development F2 as the
no-feedback baseline (`0.608053`). The next larger RM3 profile reached reranker
weight loading only at `43212` seconds, approximately the Kaggle 12-hour limit;
the supplied log contained no OOM traceback. The notebook was then changed to
screen six Rocchio settings. On 2026-10-09 the owner reported that this Rocchio
screen also stalled before producing a usable comparison; no completed Rocchio
metric or traceback was supplied, so no effectiveness or OOM claim is made.
RM3 and Rocchio are therefore deferred, remain disabled by default, and are not
part of the selected pipeline. They should be reconsidered only after profiling
and bounding retrieval runtime independently of the reranker. These execution
failures do not establish that pseudo-relevance feedback is ineffective; they
establish that the present implementations are not suitable for the shared
Kaggle runtime budget.

## Configuration

Meaningful experiments should use explicit YAML configuration. For example:

```yaml
experiment_name: hybrid_reranker_v1
seed: 42
method: hybrid_reranker
retrieval:
  sparse: {enabled: true, top_k: 200}
  dense: {enabled: true, model: MODEL_NAME, top_k: 200}
fusion:
  method: rrf
reranking:
  enabled: true
  model: MODEL_NAME
  candidate_pool_size: 200
  top_k: 50
evaluation:
  output_k_values: [1, 2, 3, 4, 5]
  recall_at_k: [1, 5, 10, 20, 50, 100]
```

BM25+ is available as an isolated sparse-scoring ablation through
`retrieval.bm25.variant=bm25_plus` and `retrieval.bm25.delta`. It adds the
configured lower-bound delta only for query terms present in a document; the
default remains standard BM25. Hold tokenization, `k1`, `b`, dense retrieval,
WRRF, reranker checkpoint, candidate budgets, and final policy fixed when
comparing the variants. BM25+ changes sparse ranks and therefore may change the fused
direct-chunk candidate set; report candidate Recall@k as well as final F2.

`configs/sparse_screening.yaml` and
`notebooks/archive/kaggle_sparse_bm25_plus_screening.ipynb` implement sparse tuning and
the matched BM25+ control. The predefined BM25 grid crosses `k1` values
`0.6/0.9/1.2/1.5` with `b` values `0.4/0.6/0.75`; BM25+ remains fixed at
`k1=1.2`, `b=0.75`, and delta `1.0`. Every case uses the same seed-42 split,
VnCoreNLP tokens, and top-100 depth. Selection prioritizes Recall@100, then
Recall@50, then fixed top-3 F2 because this sparse branch supplies hybrid
candidates. The notebook reports fixed output k=1/2/3/4/5/10 and candidate
Recall@1/5/10/20/50/100, verifies both underlying index caches, and packages
the BM25+ cache independently from the tuning archive. The grid winner is a
development-selected configuration and requires downstream confirmation.

The project owner reported the completed 438-query run on 2026-10-27. These
values are measured notebook outputs pending inspection of the downloaded
archive. Standard BM25 reached its best fixed-output F2 at k=3 (`0.379908`),
while BM25+ also peaked at k=3 (`0.371605`), a decrease of `0.008303` absolute.
BM25+ improved only the earliest cutoff: at k=1, precision/recall/F2 changed
from `0.374429`/`0.323386`/`0.332450` to
`0.383562`/`0.329093`/`0.338713`. At every candidate cutoff after rank 1 it was
worse:

| Method | Output k | Precision | Recall | F2 |
|---|---:|---:|---:|---:|
| BM25 | 1 | 0.374429 | 0.323386 | 0.332450 |
| BM25 | 2 | 0.251142 | 0.429740 | 0.376229 |
| BM25 | 3 | 0.196347 | 0.495782 | **0.379908** |
| BM25 | 4 | 0.165525 | 0.549815 | 0.375473 |
| BM25 | 5 | 0.142466 | 0.584442 | 0.360663 |
| BM25 | 10 | 0.085160 | 0.674837 | 0.282966 |
| BM25+ delta 1.0 | 1 | 0.383562 | 0.329093 | 0.338713 |
| BM25+ delta 1.0 | 2 | 0.244292 | 0.415226 | 0.364252 |
| BM25+ delta 1.0 | 3 | 0.192542 | 0.484176 | **0.371605** |
| BM25+ delta 1.0 | 4 | 0.160388 | 0.527555 | 0.361872 |
| BM25+ delta 1.0 | 5 | 0.139269 | 0.566368 | 0.351053 |
| BM25+ delta 1.0 | 10 | 0.084018 | 0.662225 | 0.278669 |

| Sparse method | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.323386 | **0.584442** | **0.674837** | **0.754120** | **0.834758** | **0.877452** |
| BM25+ delta 1.0 | **0.329093** | 0.566368 | 0.662225 | 0.738519 | 0.814264 | 0.853783 |

The corresponding BM25+ minus BM25 candidate-recall differences are
`+0.005707`, `-0.018074`, `-0.012612`, `-0.015601`, `-0.020494`, and
`-0.023669` at @1/5/10/20/50/100. The verified BM25+ cache was packaged as
`retrieval_cache_bm25_plus_vncorenlp_delta1.tar` (reported size 0.0721 GiB),
and the completed tuning archive as
`sparse_tuning_bm25_and_bm25_plus.tar.gz` (reported size 0.01466 MiB). This
standalone result does not prove that BM25+ cannot alter
WRRF complementarity beneficially, but it does not support replacing standard
BM25 as the default sparse branch. Any end-to-end BM25+ run remains an isolated
secondary ablation with all dense, reranker, and final-selection settings fixed.

The same reported run completed the predefined 12-case standard-BM25 grid. The
table is ordered by the declared selection key: Recall@100, then Recall@50,
then fixed top-3 F2.

| k1 | b | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | F2@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.2** | **0.75** | 0.323386 | **0.584442** | 0.674837 | **0.754120** | 0.834758 | **0.877452** | 0.379908 |
| 1.5 | 0.75 | 0.314253 | 0.570363 | 0.673125 | 0.752979 | **0.838182** | 0.877071 | **0.381754** |
| 1.5 | 0.60 | 0.293705 | 0.570820 | 0.675027 | 0.740422 | 0.836660 | 0.868319 | 0.367842 |
| 0.9 | 0.75 | 0.319961 | 0.578354 | **0.681061** | 0.751837 | 0.828289 | 0.867939 | 0.373696 |
| 1.2 | 0.60 | 0.297130 | 0.573483 | 0.677310 | 0.747652 | 0.834377 | 0.863753 | 0.374232 |
| 0.6 | 0.75 | **0.326810** | 0.564085 | 0.671929 | 0.744227 | 0.814590 | 0.859491 | 0.370638 |
| 1.5 | 0.40 | 0.287997 | 0.538992 | 0.642248 | 0.729006 | 0.826006 | 0.859187 | 0.353890 |
| 0.9 | 0.60 | 0.305121 | 0.571581 | 0.668504 | **0.754120** | 0.821059 | 0.859187 | 0.374622 |
| 1.2 | 0.40 | 0.292564 | 0.550408 | 0.641868 | 0.732050 | 0.816873 | 0.856143 | 0.360296 |
| 0.9 | 0.40 | 0.287997 | 0.562394 | 0.643009 | 0.725582 | 0.808121 | 0.855001 | 0.368070 |
| 0.6 | 0.60 | 0.302838 | 0.563780 | 0.663748 | 0.738139 | 0.817689 | 0.852261 | 0.377485 |
| 0.6 | 0.40 | 0.296369 | 0.556306 | 0.639775 | 0.714547 | 0.796380 | 0.848076 | 0.372170 |

The selected `k1=1.2`, `b=0.75` configuration is therefore unchanged. The
closest alternative, `1.5/0.75`, improves Recall@50 by `0.003424` and F2@3 by
`0.001846`, but reduces the primary Recall@100 metric by `0.000381`. This grid
supports retaining the existing Lucene-style defaults for the main hybrid run;
the small trade-off between the top two settings should not be overstated
without paired per-query or downstream hybrid confirmation.

The optional `scripts/evaluate_lightgbm_fusion.py` performs a low-cost first
test of learned feature fusion from an existing development run. It consumes
saved reranker, first-stage, sparse, dense, and chunk-candidate artifacts and
uses query-grouped five-fold out-of-fold pointwise LightGBM predictions. This
avoids fitting a candidate's score with its own label. The pooled OOF threshold
is still selected and reported on development data, so it remains a diagnostic,
not an unbiased test estimate.

For the active pointwise reranker experiment, LightGBM is a post-reranking
selector rather than part of checkpoint selection. First choose the pool-50 or
pool-100 checkpoint using the matched fixed-top-2 development gate. Then fit
LightGBM only for the selected finalist. `base_v1` preserves the historical
15-feature base for the current two-dense configuration, `enhanced_v1` preserves
the 41-feature comparison, and `enhanced_v2`/`enhanced_v3` are the new 61- and
69-feature treatments. New reranking runs persist each scored chunk's reranker
logit, input rank, length, lexical overlap, and citation overlap, so v2/v3 fail
closed rather than silently imputing these fields from old artifacts. The v3
article-static fields are generated deterministically from the same flattened
corpus with `scripts/build_ltr_static_features.py` and may be uploaded as a
small Kaggle Input. Report both split and gain importance, but decide removal
with query-grouped OOF feature-group ablations; a low importance value from one
fit is not sufficient evidence to remove a feature. These schemas are
implemented treatments, not measured improvements; their dev results must be
recorded before choosing one for test inference.

`scripts/run_lightgbm_fusion.py` extends that diagnostic into a submission
handoff. It first reports the same query-grouped OOF result, then fits one final
pointwise classifier on all labeled development candidate rows and applies it
to saved public/private epoch-3 candidate artifacts. It packages fixed top-2
and the OOF-selected threshold with empty-only top-2 fallback. The final fit is
appropriate for generating test predictions after selecting the treatment, but
does not make the pooled OOF threshold unbiased. Submission inference must save
the reranker ranking, unchanged first-stage ranking, sparse/dense component
rankings, and fused chunk candidates for every test split so training and test
feature schemas remain identical.

The selected final-set policy is available through
`--adaptive-selection score_gap`. For each query it examines score drops after
ranks 1 through `maximum_k - 1`, returns the prefix ending at the largest drop
when that drop exceeds a tuned minimum, and otherwise uses a fixed fallback
prefix. The tuner searches exact observed gap values plus the fixed-fallback
case and records `minimum_gap`, `fallback_k`, and `maximum_k`. This policy does
not change ranking, retrieval, reranking, or the two baseline ZIPs. Omitting the
flag (the default `disabled`) exactly preserves the original top-2 and global-
threshold handoff. Keep the flag explicit so the older policies remain
reproducible controls.

Post-freeze diagnostics also report fixed reranker and LightGBM cutoffs at
1/2/3/4/5/10/20/30/50. These are views of the same saved rankings and do not
change the primary final policy. `scripts/evaluate_lightgbm_fusion.py` compares
LightGBM input depths 10, 20, 30, and all reranked article rows; each depth is
fit and evaluated independently with the same query-grouped folds. Score-gap
tuning now permits `maximum_k` from 3 through 10 and reports a prefix-at-10
oracle. Treat the depth matrix and wider score-gap search as post-freeze
selector ablations: select with nested OOF, preserve the previous all-row,
maximum-5 score-gap result as the historical control, and do not use public or
private scores to choose among them.

Report three score-gap views separately: pooled OOF tuning supplies the frozen
deployment parameters; nested OOF evaluation estimates transfer by generating
inner-OOF scores and tuning only within each outer-training fold; and the
per-query best prefix through rank 5 is labeled a non-deployable oracle upper
bound. The oracle is only a headroom diagnostic. The aligned result below is
now complete; report its nested-OOF and frozen public/private results rather
than treating the more optimistic oracle as measured deployable performance.

### Epoch-3 LightGBM fusion result

The project owner reported the completed five-fold query-grouped OOF run on
2026-10-07. It used the unchanged epoch-3 direct-chunk candidate artifacts and
the thirteen documented reranker, first-stage, sparse, dense, chunk, and query-
length features. The original comparison selected threshold `0.1020920005`
with empty-only top-2 fallback; the subsequently completed score-gap treatment
is included here as the selected final-set policy:

| System | Policy | Precision | Recall | F2 |
|---|---|---:|---:|---:|
| LightGBM nested OOF | score-gap | 0.517884 | 0.682344 | **0.641595** |
| LightGBM pooled OOF | score-gap | 0.520548 | 0.683486 | 0.643219 |
| Epoch 3 | threshold + fallback 2 | 0.487123 | 0.648288 | 0.608053 |
| LightGBM OOF | threshold + fallback 2 | 0.484078 | 0.683295 | **0.631332** |
| Epoch 3 | fixed top 2 | 0.365297 | 0.614992 | 0.541029 |
| LightGBM OOF | fixed top 2 | 0.375571 | 0.629833 | **0.554723** |

The OOF threshold treatment improves F2 by `0.023279` absolute while increasing
recall by `0.035007` and reducing precision by `0.003045`. Its mean output size
is `2.137` articles (median 2, minimum 1, maximum 7). The fixed-top-2 gain of
`0.013694` shows that the learned ordering improves independently of threshold
selection. These are development model-selection measurements; the pooled OOF
threshold remains selected on the same 438 queries.

The owner then reported Codabench scores of private F2 `0.5882` for the frozen
OOF threshold policy, private F2 `0.5397` for fixed top 2, and public F2 `0.6160`
for the frozen threshold policy. Relative to the previously reported frozen
epoch-3 private score `0.5589`, LightGBM improves private F2 by `0.0293`
absolute. These leaderboard values are user-reported pending inspection of the
saved LightGBM provenance, predictions, ZIP payloads, and finished submission
records. They are single frozen-policy evaluations, not permission to retune
the threshold on private results.

On 2026-10-26, the owner additionally reported Codabench F2 `0.6187` for the
public score-gap submission and `0.5954` for the private score-gap submission.
Against the previously submitted LightGBM threshold policy, these are gains of
`0.0027` public and `0.0072` private. The direction agrees with the completed
nested-OOF score-gap estimate (`0.641595` versus pooled-OOF threshold
`0.631332`), so this is evidence that query-adaptive final-set selection
transferred rather than merely improving pooled development tuning. These
leaderboard values remain owner-reported pending inspection of the saved
artifacts and finished submission records; they must not be used to retune the
policy on the private set.

Mean split-count importance ranked reranker score and margin first, followed by
sparse reciprocal rank, dense score, sparse score, and chunk fusion score. This
supports complementary retrieval signals but is not a causal attribution;
split-count importance is biased by feature scale/cardinality and should be
supplemented by ablation or permutation importance before making a feature-
specific claim.

### Multi-dense LightGBM result

On 2026-10-27, the owner reported the completed BM25 + VietLegal-Harrier +
multilingual-E5 treatment with fixed WRRF weights `0.1/0.5/0.4`, epoch-3
reranking, and a newly trained fifteen-feature LightGBM model. The development
set contained 438 queries, 13,973 candidate rows, 509 positive rows, and 13,464
negative rows. Its principal measured development results were:

| System | Policy | Precision | Recall | F2 |
|---|---|---:|---:|---:|
| LightGBM pooled OOF | score-gap, max 3 | 0.561644 | 0.694597 | **0.663198** |
| LightGBM nested OOF | score-gap, max 3 | 0.566210 | 0.689650 | **0.660836** |
| LightGBM OOF | threshold + fallback 2 | 0.435151 | 0.737519 | 0.647531 |
| Epoch 3 | threshold + fallback 2 | 0.483893 | 0.629833 | 0.594003 |
| LightGBM OOF | fixed top 2 | 0.388128 | 0.653501 | 0.574888 |
| Epoch 3 | fixed top 2 | 0.355023 | 0.603196 | 0.529209 |

The pooled score-gap policy used minimum gap `0.430864`, fallback 3, maximum 3,
and mean output size `1.982`; its nested-OOF estimate was only `0.002362` lower,
which indicates limited policy-selection optimism on this development split.
The LightGBM threshold policy selected probability `0.075641` with empty-only
fallback 2 and mean output size `2.614` (maximum 11).

The owner also reported the following finished Codabench F2 values:

| Retrieval/model | Final policy | Public | Private |
|---|---|---:|---:|
| Multi-dense LightGBM | threshold + fallback 2 | **0.6384** | **0.5952** |
| Multi-dense LightGBM | score-gap | 0.6260 | 0.5920 |
| Single-BGE LightGBM | threshold + fallback 2 | 0.6160 | 0.5882 |

The later full `packed_stream` batch-256 rerun produced the following completed
Codabench results on 2026-10-28:

| Submission ID | Policy | Split | F2 |
|---:|---|---|---:|
| 1014193 | LightGBM threshold + fallback 2 | public | 0.6293 |
| 1014192 | LightGBM threshold + fallback 2 | private | **0.5960** |
| 1014187 | LightGBM score-gap | public | 0.6228 |
| 1014186 | LightGBM score-gap | private | 0.5914 |

Relative to the earlier matched multi-dense submissions, threshold changed by
`-0.0091` public and `+0.0008` private, while score-gap changed by `-0.0032`
public and `-0.0006` private. Threshold still exceeded score-gap by `0.0065`
public and `0.0046` private in the new run. These leaderboard observations are
reported outcomes, not a license to tune further on the private split.

Under the matched threshold policy, multi-dense improves the reported public
score by `0.0224` and private score by `0.0070` over single BGE. This is evidence
that the additional dense signals transfer, although the private gain is
modest. The development ordering between final-set policies did not transfer:
score-gap led threshold by `0.015667` development F2, but threshold led
score-gap by `0.0124` public and `0.0032` private. Leaderboard results expose
only aggregate F2 here, so they cannot establish whether this reversal arose
from output-size shift, calibration shift, or ranking errors. These submissions
must be recorded as completed evaluations, not used to tune another policy on
the private set.

### Full multi-dense query failure analysis

The 438-query packed-stream batch-256 archive contains 593 labeled gold
articles and complete component, fusion, direct-chunk, epoch-3, and OOF
LightGBM rankings. Measured macro Recall@50 was `0.834758` for BM25S,
`0.924364` for VietLegal-Harrier, `0.896015` for multilingual-E5,
`0.922081` after article-level fusion, and `0.903033` for the actual top-50
direct-chunk candidate pool consumed by reranking. The top-50 union oracle of
the three components was `0.942901`; it is descriptive and not a realizable
fusion result.

Of 593 gold articles, 509 survived into the direct-chunk candidate pool. The 84
missing gold articles decomposed as follows: 36 were outside every component's
top 100; 16 were outside every component's top 50 but appeared in at least one
component at ranks 51--100; 15 were found by a component at top 50 but fusion
placed them at ranks 51--100; two were found by a component but excluded from
the fused top 100; and 15 appeared in fused article top 50 but not in the
deduplicated direct-chunk top-50 pool. This identifies three separate controls
for future work: component recall, fusion, and article-to-chunk candidate
construction.

The archived handoff contained exactly 50 chunks for every query, but those
chunks covered only `31.9041` unique article IDs per query on average (minimum
14, median 32, maximum 49); all 438 queries had fewer than 50 unique article
IDs. Thus the fixed 50-pair budget was partly spent on multiple passages from
the same article. This is a verified property of the saved candidate artifact,
not an inference from final rankings.

`retrieval.hybrid.chunk_candidate_selection` now exposes a controlled
candidate-construction ablation. `top_chunks` is the default and preserves the
measured baseline exactly. `best_per_aid` applies the same chunk-level WRRF
scores and deterministic ordering, then retains only the first (highest-ranked)
chunk for each `aid` until the same `candidate_pool_size` is filled or the
retrieved depth is exhausted. It changes neither component retrieval, WRRF
weights, cross-encoder, nor LightGBM, and it does not claim to reproduce the
article-level fused top 50. Compare the two policies with the same depth and
50-pair budget before implementing the stronger article-aligned handoff.

Harrier supplied 14 gold articles found by no other top-50 component, while E5
and BM25S each supplied five. Harrier and E5 rescued respectively 60 and 51
BM25S top-50 misses. The principal weakness was multi-article supervision:
fused Recall@50 was `0.971787` for 319 one-gold queries, `0.809783` for 92
two-gold queries, and `0.681818` for 22 three-gold queries. The remaining five
development queries had respectively four golds (three queries, Recall@50
`1.0`), five golds (one query, `0.8`), and seven golds (one query, `0.571429`).
These tiny 4+ strata do not support a monotonic trend claim. Across all 2,190
training rows after stable per-query deduplication, gold-count frequencies were
1: 1,646; 2: 416; 3: 84; 4: 30; 5: 6; 6: 3; 7: 3; 8: 1; and 9: 1. Thus the
dataset maximum is nine, while the fixed development split maximum is seven.
Grouped by legal
source, the 77 multi-gold queries whose gold articles belonged to one law had
Recall@50 `0.801793`, while 42 cross-law multi-gold queries had `0.765079`.
Among missing gold articles, 35 had another retrieved gold article from the
same law; eight were immediately adjacent in corpus article order and 14 were
within five articles. These observations motivate a controlled same-law
structure/reference expansion, but do not by themselves prove that adjacency
or references are causal.

An additional same-law audit over all 119 multi-gold development queries found
77 whose complete gold set was in one law, 89 with at least one same-law gold
pair, 30 with an adjacent same-law pair, 39 with a same-law pair at distance at
most two, and 58 at distance at most five. The median distance among same-law
gold pairs was three articles. Therefore unrestricted neighbor expansion is not
justified: adjacency describes only one quarter of multi-gold queries, and some
4+ examples span several laws or twenty article positions.

A retrospective fixed-budget simulation nevertheless identifies one small
candidate treatment worth executing. Starting from fused article top 50,
expanding only the top five seeds by one article on either side, limiting the
expansion to four distinct aids, and removing the same number of lowest-ranked
baseline aids recovered 12 previously missing golds while dropping one former
top-50 gold. Its descriptive macro Recall@50 was `0.936867`, versus `0.922081`
for the saved baseline, and multi-gold Recall@50 was `0.818047`. The unbudgeted
version added four aids per query on average and recovered the same 12 golds.
These values were selected after inspecting the development labels and are
diagnostic upper evidence, not an unbiased measured treatment result. Any
implemented neighbor policy must remain independently disableable and be
compared at the same candidate budget before reranking.

The implemented control is `retrieval.hybrid.same_law_neighbor_expansion` and
is disabled by default. When enabled, it derives article neighbors from corpus
order within each exact `law_id`, takes distinct seed aids from the selected
chunk pool, and can insert only neighbor aids for which the configured dense
retrieval depth already supplied chunk evidence. It replaces the same number
of tail chunks, so reranker pair count remains fixed. The default treatment
parameters are seed top five, radius one, and at most four expansions. Each
accepted edge records qid, seed aid, neighbor aid, and signed distance in the
retrieval feedback/provenance artifact. This available-chunk constraint means
the executed treatment may recover fewer items than the retrospective
article-level simulation and must be reported separately.

At rank five, 374 gold articles were found by OOF LightGBM. Another 122 were in
the candidate pool but outside epoch-3 top five, while 13 were in epoch-3 top
five but outside LightGBM top five. Across all 509 candidate gold articles,
epoch-3 improved 156 relative to candidate order, left 210 unchanged, and
worsened 143; LightGBM improved 170, left 258 unchanged, and worsened 81. Query
length was a weak descriptive separator (`0.927651` fused Recall@50 for 269
queries of at most 20 whitespace tokens versus `0.911647` for 166 queries of
21--40 tokens). The current explicit-reference detector matched no development
query, so this run provides no evidence about that subgroup and the detector
must not be used to claim that explicit citations are absent in a broader
dataset.

The smallest justified next experiments are therefore: (1) compare direct
chunk top 50 against a candidate pool that guarantees inclusion of fused
article top 50; (2) test a fixed-budget union or learned fusion treatment
against WRRF without changing reranking; and (3) add deterministic, independently
disableable same-law structural or explicit-reference expansion, reporting
single-gold, same-law multi-gold, and cross-law multi-gold strata. Increasing
all candidate budgets or adding unrestricted query expansion would confound
these failure modes and is not the first recommended treatment.

The subsequent full-corpus reference audit found a stronger low-volume signal
than unrestricted neighbor expansion: outgoing same-instrument references from
top-five fused seeds provisionally recovered 18 missing top-50 gold aids while
adding 0.71 candidates per query, or 16/0.69 after excluding every source law
with an out-of-range local reference. Exact cross-document identifiers added
only one recovery at this depth. Because corpus records omit their own article
number, these edges are position-derived and remain unverified; the detailed
coverage and ambiguity counts are recorded in `docs/dataset.md`. The next
reference step is therefore a deterministic extractor with explicit
`resolved`, `position_derived`, `ambiguous`, and `missing` statuses plus a fixed
human-review sample. Only reviewed/accepted edges may enter the controlled
recall treatment.

The precision-first reference graph v2 was subsequently evaluated against the
saved packed-stream batch-256 first-stage artifact, without rerunning retrieval.
The deterministic review sample uses seed 47 and contains 105 mentions: 25
eligible same-document edges, 25 eligible cross-document edges, and stratified
unresolved/heading/amendment controls. A contextual review by Codex found all
50 sampled eligible edges consistent with their stored source text. This is a
documented assistant review, not independent human annotation, and therefore
does not close the precision gate by itself.

The retrospective fixed-split recall gate used fused top 50, seed depth five,
at most four added aids, and a fixed 50-aid budget. Baseline macro Recall@50 was
`0.922081`. Same-document edges expanded 173/438 queries, added 0.795 aids per
query on average, recovered 22 missing gold pairs, dropped one tail gold, and
produced Recall@50 `0.943227`. Cross-document edges expanded 19 queries,
recovered one gold without dropping another, and produced `0.923222`. Their
union recovered 23, dropped one, and produced `0.944368`. All 23 recovered
edges were context-inspected and matched the cited gold provision; the one
dropped gold resulted from fixed-budget tail replacement. These figures use
development labels and are diagnostic upper evidence, not an unbiased final
experiment. The detailed qid, seed edge, context, recovery, and drop records
are retained beside the graph artifact.

The graph treatment is now implemented at the same chunk-candidate boundary as
the same-law neighbor treatment and is disabled by default. It loads only the
eligible-only `reference_edges.jsonl` artifact, fails if an enabled graph is
missing or malformed, and permits `same_document` and `cross_document` scopes
to be selected independently. From the first five distinct selected article
IDs it follows outgoing references in stable artifact order, admits a target
only when that target already has a chunk in the frozen dense depth, and
replaces at most four tail candidates so the reranker budget remains fixed.
Every accepted expansion records qid, seed aid, target aid, scope, and target
law ID. Consequently, the executed treatment can be weaker than the
retrospective article-level gate and requires its own end-to-end measurement.

For graph v3, expansion selection no longer takes the first eligible edges in
raw graph-file order. It gathers every available target reachable from the top
five distinct seed aids, orders targets by their existing base chunk rank, and
uses seed order, edge order, then target aid as deterministic tie-breakers.
This prevents a long coordinated article list from consuming all four slots
merely because it appeared first in the corpus. Feedback now records each
promoted target's `base_rank`. Targets must still exist in the frozen dense
depth, and an equal number of tail candidates is replaced, preserving the
candidate budget and reranker pair count.

The first controlled matrix is: baseline; same-document references only; and
cross-document references only. Keep `top_chunks`, dense depth 100, candidate
pool 50, seed depth five, maximum four expansions, reranker, LightGBM features,
and final policy fixed. Do not combine the two reference scopes, same-law
neighbors, or `best_per_aid` until their independent effects are measured.
The 50/50 contextual edge review and inspection of all 23 diagnostic recoveries
are sufficient to begin this fail-closed ablation, but do not establish a
perfect graph. Before a final scientific claim, add an independent review of a
second fixed random sample and inspect every recovered or dropped gold edge in
each measured treatment; resolving all ambiguous mentions is not required.

### Distinct-article chunk-candidate ablation

The 2026-10-28 matched `best_per_aid` run is a negative ablation. Its saved
split, first-stage predictions, and component rankings are byte-identical to
the `top_chunks` packed-stream batch-256 baseline; the reranker checkpoint,
FP16 mode, candidate budget 50, retrieved chunk depth 100, LightGBM protocol,
and seed 42 also match. Thus the measured difference is attributable to chunk
candidate selection rather than a changed retrieval run.

`best_per_aid` increased the number of distinct candidates from a mean of
31.90 to exactly 50 aids per query and improved candidate Recall@50 from
`0.903033` to `0.926647`. It added 18 gold candidate pairs and removed none.
This coverage gain did not survive evidence reranking: epoch-3 Recall@5 fell
from `0.725913` to `0.673782`, with 7 query-level wins and 39 losses, and its
best threshold F2 fell from `0.594003` to `0.546620`. LightGBM OOF Recall@5
fell from `0.787633` to `0.754642`; nested OOF score-gap F2 fell from
`0.657352` to `0.625154`, while pooled diagnostic score-gap F2 fell from
`0.660765` to `0.631653`. The loss was particularly large for three-gold
queries at LightGBM top five: recall fell from `0.469697` to `0.333333` over
22 such queries.

The cause is the mismatch between dense-fusion chunk selection and
cross-encoder evidence selection. The control retained a mean of 18.10 extra
chunks per query from already represented aids, allowing the cross-encoder's
article-level max aggregation to choose among several passages. Of 509 gold
article-query pairs present in both candidate pools, 417 had multiple chunks
in the control. `best_per_aid` kept the same first/best-fusion chunk but removed
all alternatives before the cross-encoder, and the LightGBM `chunk_count`
feature consequently had zero importance. Broader article coverage therefore
traded away more valuable within-article evidence coverage.

Keep `top_chunks` as the primary baseline. Do not use strict one-chunk-per-aid
for the current cross-encoder. If this direction is revisited, test a bounded
hybrid allocation such as two chunks for high-ranked aids plus remaining slots
for distinct articles, or rerank a larger chunk pool before article-level max;
either is a new controlled ablation rather than a replacement justified by the
present result.

The next WRRF tuning run keeps this restored `top_chunks` baseline and screens
40 inexpensive offline combinations from the already saved component
rankings: `rrf_k` in `10/20/40/60` crossed with ten normalized weight triples.
The grid retains the measured BM25/Harrier/E5 control `0.10/0.50/0.40`, spans
BM25 weights `0.05--0.20`, and concentrates Harrier weights at `0.50--0.65`
while retaining equal-dense and E5-heavier controls. This concentration is
motivated by the previous LightGBM importance observation but does not treat
feature importance as causal evidence. Selection maximizes first-stage
Recall@50, breaking ties with Recall@20, top-3 F2, and declaration order. The
weight screen itself uses component article rankings and is therefore
unaffected by chunk-boundary expansion. The selected reranker run enables only
same-document explicit-reference expansion; same-law neighbors and
cross-document references remain off. Because selection and the subsequent
scoreboard use the same development split, the latter is a tuning diagnostic;
the selected setting must be frozen before a test comparison or repeated-split
robustness check.

A local preflight of this exact grid against the saved byte-identical baseline
component rankings selected `rrf_k=10` with BM25/Harrier/E5 weights
`0.15/0.55/0.30`: Recall@50 `0.927789`, Recall@20 `0.875826`, and Recall@100
`0.952794`. The measured `rrf_k=20`, `0.10/0.50/0.40` control had Recall@50
`0.922081`. The apparent `+0.005708` development Recall@50 gain is a
retrospective tuning result, not yet evidence of downstream F2 improvement;
the Kaggle run must freeze this selection before reranking and report the full
epoch-3 and nested-OOF LightGBM comparison.

For the final report's controlled first-stage component table, rerun BM25,
Harrier, E5-large, BM25+Harrier, Harrier+E5-large, and the full three-component
hybrid from the same saved rankings. Use WRRF `k=10` throughout. Single
components have unit weight; subset combinations inherit and renormalize the
frozen `0.15/0.55/0.30` ratio and use the declared rounded controls
BM25/Harrier `0.21/0.79` and Harrier/E5 `0.65/0.35`. This is a component
ablation, not six independently tuned systems. Report fixed-top-2 metrics and
Recall@50 as the main candidate metric because the downstream reranker consumes
50 articles; retain Recall@100 only as an artifact diagnostic.

This notebook run changes both WRRF and reference treatment relative to the
measured baseline, so its final delta cannot identify their separate causal
effects. It is an operational combined candidate for the owner, while the
already documented matched reference-only run remains necessary for a clean
scientific ablation.

Split-count feature importance ranked reranker score (`309.6`) and reranker
margin (`278.2`) first. Harrier raw score (`212.6`) and reciprocal rank (`164.0`)
both exceeded the corresponding multilingual-E5 values (`174.0`, `97.4`). This
is consistent with useful Harrier signal but is not a causal component
ablation; correlated features and split-count bias prevent that interpretation.

Record model identifiers, candidate pool size, final count, thresholds, and
normalization or fusion parameters whenever applicable.

`method` is the top-level strategy selector. The experiment runner should expose
one stable command such as:

```bash
python scripts/run_experiment.py --config configs/bm25.yaml
python scripts/run_experiment.py --config configs/organizer_two_stage.yaml
```

Kaggle paths can be supplied without changing the tracked config:

```bash
python scripts/run_experiment.py \
  --config configs/bm25.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --train-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --output-root /kaggle/working/results
```

Each method registers a pipeline implementation behind a common interface.
Method-specific options stay in their own config section, while dataset,
runtime, output, seed, and evaluation settings remain shared. Adding a method
must not require another top-level runner or duplicate data/evaluation logic.

CLI overrides may support quick controlled trials, for example
`--set retrieval.top_k=100`, but the resolved configuration must always be saved
with the result. Reject unknown method names and unknown config keys instead of
silently ignoring them.

Hardware settings must also be explicit rather than inferred from the current
machine. GPU-capable stages should accept device and worker-count settings. The
preferred Kaggle profile targets two NVIDIA T4 GPUs, but the same experiment
must retain CPU and single-GPU modes for validation and debugging.

## Prediction format

For internal experiment artifacts, use JSONL or Parquet and preserve at least:

```json
{"qid": 123, "retrieved_aids": [1001, 845, 9002], "scores": [12.3, 11.7, 10.4]}
```

Retain meaningful intermediate scores when useful for analysis.

## Official submission format

The Codabench submission payload is a JSON array named `results.json`:

```json
[
  {"qid": 1, "relevant_laws": [100, 101, 102]},
  {"qid": 2, "relevant_laws": [200, 201, 202]}
]
```

Here `qid` corresponds to the query identifier (the inspected training Parquet
uses `qid`; the task-page example used `id`), and `relevant_laws` contains
predicted article IDs (`aid`). Scores and other internal fields must not appear
in the official payload unless the organizer changes the schema.

Package the file as `results.zip` with exactly one member at the ZIP root:
`results.json`. It must not be nested in a directory. Before packaging, validate
that every expected query appears exactly once, no query ID is missing, and the
JSON shape and types match the required schema.

The published participation rules state:

- at most 10 submissions per team per day;
- at most 5 submissions per user in the Private Phase;
- the username should represent the team;
- an official final result requires a working-notes paper describing the method;
- non-compliant submissions may be disqualified by the organizer.

Treat submission budgets as evaluation safeguards: do not tune against the
private leaderboard, and choose Private Phase submissions deliberately.

## Run artifacts

A meaningful run should normally produce:

```text
results/<experiment_name>/
├── config.yaml
├── metrics.json
├── metrics_by_output_k.json
├── metrics_by_score_threshold.json  # reranker when enabled
├── predictions.jsonl
└── run_info.json
```

`run_info.json` should capture the experiment name, timestamp, seed, Git revision
and dirty state, model identifiers, dataset/split identifiers, and hardware and
runtime where relevant.

Large intermediate artifacts remain outside Git. Selected small configs,
metrics, and provenance files may be force-added after review when they preserve
a meaningful result.

## Research record

Important decisions, verified dataset statistics, measured results, limitations,
and changes to the experimental protocol must also be recorded in the relevant
Markdown document. Result artifacts remain the machine-readable source for a
specific run, but important findings must not exist only in ignored artifacts or
conversation history.

Use these destinations:

- dataset observations and preprocessing decisions: `docs/dataset.md`;
- metric definitions and evaluator behavior: `docs/evaluation.md`;
- method choices, ablations, and research direction: `docs/research_plan.md`;
- configurations, measured runs, and artifact conventions:
  `docs/experiments.md`;
- local/Kaggle/runtime decisions: `docs/execution.md`.

Always distinguish values measured by this repository from values published by
the organizer or other teams.

### Organizer baseline configuration status

The organizer's extended paper establishes `output_k=2` for the complete
two-stage baseline and reports top-1/top-2/top-5 ablations. It does not reveal
the BM25 candidate-pool depth used before reranking. Accordingly,
`configs/organizer_two_stage.yaml` records final `output_k: 2` but remains a
non-runnable specification until the missing model and training choices are made
explicit. `configs/bm25.yaml` is this repository's reproducible BM25 baseline,
not a claim of exact organizer-code reproduction.

## Recorded baseline runs

### BM25, seed-42 development split

Measured locally on 2026-09-28 using `configs/bm25.yaml`:

- corpus: 59,628 indexed non-blank articles;
- split: deterministic 20% development split, 438 of 2,190 queries;
- tokenizer: lowercase Unicode NFKC word tokens;
- BM25: `k1=1.2`, `b=0.75`, retrieval `top_k=100`;
- selected output size: `output_k=3`;
- Precision: `0.1803652968`;
- Recall: `0.4525929550`;
- F2: `0.3476503328`;
- candidate Recall@1: `0.2673135464`;
- candidate Recall@5: `0.5267938682`;
- candidate Recall@10: `0.6378723636`;
- candidate Recall@20: `0.7206349206`;
- candidate Recall@50: `0.8040661013`;
- candidate Recall@100: `0.8441563383`;
- latest local CPU runtime: approximately 68 seconds.

Candidate Recall@k is computed from the complete BM25 ranking up to 100, while
official Precision, Recall, and F2 use only the final top three predictions.
The values show that BM25 preserves substantially more gold evidence in its
candidate pool than survives final fixed-k selection; future reranker comparisons
must reuse or explicitly change this candidate budget.

`output_k=3` was selected on this development split after comparing fixed cutoffs,
so this is a development result rather than an unbiased final-test estimate. It
must not be compared directly with the organizer's published private-test BM25
F2 of `0.3365`.

### BM25 Vietnamese word-segmentation ablation

Reported from Kaggle runs by the project owner on 2026-09-29. These values use
the same 438-query seed-42 development split, BM25 `k1=1.2`, `b=0.75`,
candidate `top_k=100`, and final `output_k=3`, according to the run context.
“PyVi” below interprets the submitted label `pypi` as PyVi. The corresponding
Kaggle artifacts have not yet been copied into or independently inspected from
this checkout, so the values are measured but user-reported pending artifact
verification.

| Tokenization | Precision | Recall | F2 | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Regex/NFKC default | 0.180365 | 0.452593 | 0.347650 | 0.267314 | 0.526794 | 0.637872 | 0.720635 | 0.804066 | 0.844156 |
| PyVi | 0.196347 | 0.490645 | 0.377485 | 0.324908 | 0.579305 | 0.669129 | 0.742895 | **0.842803** | **0.878593** |
| Underthesea | 0.194825 | 0.491977 | 0.376981 | **0.325288** | 0.575500 | 0.673695 | **0.756023** | 0.835192 | 0.876310 |
| VnCoreNLP | **0.196347** | **0.495782** | **0.379908** | 0.323386 | **0.584442** | **0.674837** | 0.754120 | 0.834758 | 0.877452 |

Relative to the regex baseline, VnCoreNLP improves development F2 by `0.032257`
absolute (about 9.28% relative). PyVi has the highest candidate Recall@50 and
Recall@100, while VnCoreNLP has the best final top-three F2. The differences
among the three segmenters are small enough that this single split does not
establish general superiority. For the immediate pipeline, use VnCoreNLP as the
best measured final-output BM25 treatment and retain PyVi as a lightweight
candidate-recall alternative. Before final reporting, archive and inspect each
resolved config, metrics, split IDs, run metadata, and dependency versions.

### Multilingual-E5-base chunking runs

Reported from Kaggle by the project owner on 2026-09-29 for the 438-query
seed-42 development split. From the run context, the treatment is
`intfloat/multilingual-e5-base` with 512 total tokens, 64-token overlap, max
chunk-to-aid aggregation, candidate `top_k=100`, and final `output_k=3`.
Artifacts have not yet been copied into or independently inspected from this
checkout, so the configuration attribution is user-reported pending verification.

- Precision: `0.2070015221`;
- Recall: `0.5241628615`;
- F2: `0.4012165424`;
- candidate Recall@1: `0.3253424658`;
- candidate Recall@5: `0.6058056099`;
- candidate Recall@10: `0.6984072624`;
- candidate Recall@20: `0.7964666232`;
- candidate Recall@50: `0.8747662535`;
- candidate Recall@100: `0.9046586214`.

This exceeds the measured VnCoreNLP BM25 run by `0.021309` absolute F2 (about
5.61% relative) and `0.027207` absolute Recall@100. It does **not** yet isolate
the effect of chunking: the required controlled comparison is the same E5 model
and retrieval configuration with `article_truncate`. Inspect the fixed-window
`config.yaml`, `run_info.json`, split IDs, and chunk statistics before treating
the run as fully verified.

The matching article-truncation control was subsequently reported from Kaggle
on 2026-09-29 with Precision `0.1917808219`, Recall `0.4853500761`, F2
`0.3715880880`, and candidate Recall@1/5/10/20/50/100 of `0.3006088280`,
`0.5803109372`, `0.6703033268`, `0.7755164166`, `0.8481843879`, and
`0.8798488802`. Assuming the saved configs and split IDs confirm the stated
controlled setup, fixed 512/64 token windows improve F2 by `0.029628` absolute
(about 7.97% relative) and Recall@100 by `0.024810` over article truncation.

| E5 representation | Precision | Recall | F2 | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Article truncate | 0.191781 | 0.485350 | 0.371588 | 0.300609 | 0.580311 | 0.670303 | 0.775516 | 0.848184 | 0.879849 |
| Token window 512/64 | 0.207002 | 0.524163 | 0.401217 | 0.325342 | 0.605806 | 0.698407 | 0.796467 | 0.874766 | 0.904659 |
| Recursive token 512/64 | 0.204718 | 0.514650 | 0.395037 | 0.325342 | 0.613416 | 0.705121 | 0.793803 | 0.873625 | 0.905420 |
| Khoản 512/64 fallback | 0.221461 | 0.557154 | 0.427540 | 0.354262 | 0.647663 | 0.728142 | 0.786628 | 0.869684 | 0.901288 |
| Điểm + Khoản lead-in 512/64 | **0.230594** | **0.582763** | **0.446409** | **0.367199** | **0.661285** | **0.738035** | **0.796521** | **0.879273** | **0.911562** |

The improvement is present at every recorded cutoff and in both final precision
and recall. Article-truncated E5 is below the measured VnCoreNLP BM25 F2
(`0.371588` versus `0.379908`), whereas fixed-window E5 exceeds it. This supports
using fixed-window E5, not truncated E5, as the dense component entering later
hybrid experiments. This fixed-versus-truncate comparison alone does not test
recursive or legal-structure boundaries. A paired per-query analysis and
artifact verification remain required before a statistical claim.

The matching `recursive_token` 512/64 run was reported from Kaggle on
2026-09-29 with Precision `0.2047184170`, Recall `0.5146499239`, F2
`0.3950373221`, and candidate Recall@1/5/10/20/50/100 of `0.3253424658`,
`0.6134159600`, `0.7051206784`, `0.7938030007`, `0.8736247010`, and
`0.9054196564`. Relative to fixed token windows, recursive boundaries reduce F2
by `0.006179` and final recall by `0.009513`; they improve Recall@5 by `0.007610`,
Recall@10 by `0.006713`, and Recall@100 by only `0.000761`. Fixed 512/64 windows
therefore remain the primary non-structural dense baseline. Recursive splitting
is retained as an ablation showing that generic paragraph/sentence boundary
preference does not automatically improve final DRiLL effectiveness.

The matching `khoan` run was reported from Kaggle by the project owner on
2026-09-29 with Precision `0.2214611872`, Recall `0.5571537291`, F2
`0.4275400431`, and candidate Recall@1/5/10/20/50/100 of `0.3542617960`,
`0.6476625353`, `0.7281419874`, `0.7866275277`, `0.8696836269`, and
`0.9012883235`. This attribution assumes the same model, seed-42 split,
512-token budget, 64-token fallback overlap, max chunk-to-aid aggregation,
candidate `top_k=100`, and final `output_k=3`; the Kaggle artifact remains to be
copied and checked.

Under that controlled attribution, Khoản chunking improves F2 over fixed token
windows by `0.026324` absolute (about 6.56% relative), final precision by
`0.014460`, and final recall by `0.032991`. It also improves candidate Recall@1,
@5, and @10 by `0.028919`, `0.041857`, and `0.029735`, respectively. Recall@20,
@50, and @100 are lower by `0.009839`, `0.005083`, and `0.003370`. The result is
therefore evidence that legal Khoản boundaries improve early ranking and the
selected final output, not that they increase deep candidate coverage. Pending
artifact and paired per-query verification, `khoan` is the leading dense
representation before the finer Điểm treatment, while fixed 512/64 remains the
appropriate non-structural control.

The matching `diem` run was reported from Kaggle by the project owner on
2026-09-29 with Precision `0.2305936073`, Recall `0.5827625571`, F2
`0.4464089364`, and candidate Recall@1/5/10/20/50/100 of `0.3671993912`,
`0.6612850620`, `0.7380354425`, `0.7965209828`, `0.8792726680`, and
`0.9115622962`. This attribution assumes the same seed-42 split, E5 model,
512/64 budget and fallback, max chunk-to-aid aggregation, candidate `top_k=100`,
and final `output_k=3`; artifact metadata remains pending inspection.

Relative to Khoản chunking, Điểm plus Khoản lead-in improves F2 by `0.018869`
absolute (about 4.41% relative), precision by `0.009132`, and recall by
`0.025609`. Unlike the Khoản-versus-fixed comparison, it also improves candidate
recall at every recorded cutoff: +`0.012938`, +`0.013623`, +`0.009893`,
+`0.009893`, +`0.009589`, and +`0.010274` at @1/5/10/20/50/100. It exceeds the
fixed-window F2 by `0.045192` absolute (about 11.26% relative). Pending artifact
and paired per-query verification, `diem` is now the leading dense
representation and provides the strongest evidence so far that progressively
finer legal structure improves this retrieval setup.

### BGE-M3 dense runs

The project owner reported a BGE-M3 Khoản 512/64 run on 2026-09-30 with
Precision `0.2382039574`, Recall `0.5985540335`, F2 `0.4595227189`, and
candidate Recall@1/5/10/20/50/100 of `0.4077245053`, `0.6648401826`,
`0.7567569037`, `0.8311480757`, `0.8807186345`, and `0.9119428137`. Attribution
to the same seed-42 split, max chunk-to-aid aggregation, candidate `top_k=100`,
and final `output_k=3` is user-reported pending artifact inspection.

Two earlier BGE-M3 results had F2 `0.4695224772` and `0.4678786481`, but their
exact `chunking.method` attribution remains unresolved in chat. They must not be
labelled fixed-window, Điểm, or article-truncate until their saved `config.yaml`
or `run_info.json` is inspected. In particular, the previously supplied BGE-M3
commands selected `diem` explicitly. A BGE-M3 `token_window` 512/64 run is still
required as the non-structural control before attributing gains to legal
structure under the stronger encoder.

### BGE-M3 plus VnCoreNLP WRRF runs

On 2026-09-30, the project owner reported a VnCoreNLP BM25 plus BGE-M3 dense
WRRF run using sparse/dense weights `0.4/0.6` with Precision `0.2404870624`,
Recall `0.6006468798`, F2 `0.4622048314`, and candidate
Recall@1/5/10/20/50/100 of `0.3976407915`, `0.6862252664`, `0.7847249402`,
`0.8475918678`, `0.9054740161`, and `0.9307240705`. It improved deep candidate
recall but underperformed the strongest reported standalone BGE result at the
final top three.

Changing only the WRRF weights to sparse/dense `0.2/0.8` produced Precision
`0.2496194825`, Recall `0.6200532725`, F2 `0.4781414679`, and candidate
Recall@1/5/10/20/50/100 of `0.3942161339`, `0.6910958904`, `0.7754022614`,
`0.8368558382`, `0.8941128506`, and `0.9161285062`. Relative to the `0.4/0.6`
run, final F2 improves by `0.015937` even though recall at deeper candidate
cutoffs falls. Relative to the strongest earlier reported standalone BGE result
(F2 `0.4695224772`), it improves Precision by `0.005327`, Recall by `0.009893`,
and F2 by `0.008619` (about 1.84% relative). This indicates that BM25 is useful
as a low-weight complement, while a `0.4` sparse weight over-influences final
top-three ordering.

The run-directory name used in chat contained `wrrf60`, while the executed
command set `rrf_k=30`; the saved resolved config must settle this attribution.
All hybrid values remain user-reported pending inspection of `config.yaml`,
`run_info.json`, component rankings, and confirmation that the hybrid dense
ranking matches its standalone cached run.

With sparse/dense weights fixed at `0.2/0.8`, lowering `rrf_k` from 30 to 20
produced Precision `0.2511415525`, Recall `0.6257610350`, F2 `0.4819725558`, and
candidate Recall@1/5/10/20/50/100 of `0.4010654490`, `0.6918569254`,
`0.7748314851`, `0.8376168732`, `0.8941128506`, and `0.9248804088`. This is an
absolute F2 gain of `0.003831` over the reported `rrf_k=30` run and is the
current selected first-stage configuration. The value is development-tuned and
must remain fixed for reranker comparisons.

## Cross-encoder reranking

The implemented `hybrid_reranker` pipeline freezes the hybrid first stage and
reranks only its configured candidate pool. `configs/hybrid_reranker.yaml`
starts from BGE-M3 Điểm 512/64 plus BM25, WRRF `rrf_k=20`, sparse/dense weights
`0.2/0.8`, first-stage `top_k=100`, reranker candidate pool 50, and final
`output_k=3`. The untouched first-stage list is written to
`first_stage_predictions.jsonl`; BM25 and dense component lists remain in
`component_rankings.jsonl`. Candidate Recall@k continues to describe the fixed
first stage rather than the reranked list.

The initial zero-shot reranker is `BAAI/bge-reranker-v2-m3`. Candidate articles
are represented as Điểm plus their Khoản lead-in using the reranker's own
tokenizer. Documents are pre-windowed to at most 768 single-sequence tokens with
64-token overlap, then paired with the complete query under a 1,024-token pair
budget. Pair tokenization explicitly uses `truncation="only_second"`, so query
tokens are never discarded. Chunk scores are max-aggregated to `aid`; equal
scores preserve first-stage order and then ascending `aid`.

Only articles occurring in the frozen candidate pools are represented and
scored. The Hugging Face sequence-classification backend supports deterministic
CPU or one-GPU execution and `torch.nn.DataParallel` across configured CUDA
devices. FP16 autocast is enabled by config on CUDA only. Run metadata records
the reranker model/revision, resolved devices, pair and document budgets,
representation counts, structure parser statistics, aggregation, and candidate
pool size. This is zero-shot inference, not a fine-tuned DRiLL reranker.

Cross-encoder inference now defaults to `reranking.batching=global_stream`.
Query–representation pairs are assembled in deterministic query and candidate
order, scored in bounded outer groups, and then sliced back to their original
queries before the unchanged max-by-`aid` aggregation and tie-breaking rules.
This removes query boundaries from neural mini-batching: a short final batch for
one query can be filled by pairs from the next query. The legacy
`batching=per_query` path remains available as an exact parity control. The
`batching=packed_stream` path implements the same cross-query filling with a
rolling buffer bounded by the configured neural `batch_size`: queries and
candidates are visited sequentially, each scored item retains its query/rank
reference, and no corpus-wide pair list or outer group is materialized. It is
the preferred bounded-memory treatment for the 50-query, batch-128 smoke test;
parity with `per_query` is covered for article and retrieved-chunk candidates.
The
outer grouping is an orchestration bound, not the GPU mini-batch;
`outer_pair_batch_size=1024` feeds multiple calls of the configured neural
batch size. The reusable single-dense 512-token direct-chunk config retains
`batch_size=128`. A diagnostic multi-dense batch-64 run materialized and
transferred the reranker, then measured an OOM at the second neural mini-batch
(`start=64`). GPU 0 had only 355 MiB free and 14.21 GiB in use, predominantly
reported as non-PyTorch memory, while GPU 1 had 11.78 GiB free. Audit then found
that the strong CUDA/IPC resource barrier had only been connected to submission
inference; the selected development run uses `run_experiment.py` and still
called the older close-only path. Subsequent traces still found 11.3 GiB held
outside PyTorch, so the runner now uses the documented process-image handoff.
The first bounded end-to-end validation after that change completed on Kaggle
T4 x2 over 50 development queries using `packed_stream` and neural batch 128.
The reranker became ready at 761.0 seconds and reranking completed at 795.4
seconds (34.4 seconds for this smoke subset); the complete selected stage took
279.48 seconds. After reranking, the pipeline process occupied 4,306 MiB on GPU
0 and 5,988 MiB on GPU 1, with 10,603 MiB and 8,921 MiB free respectively.
This confirms that the former 11.3 GiB retrieval allocation was absent after
the process-image handoff and that packed batch-128 inference fits on two T4s.
These are engineering smoke measurements, not full-development throughput or
research metrics; the 50-query evaluation values must not be compared with the
fixed 438-query development results.
The matched batch-256 smoke run also completed over the same 50-query subset.
Measured reranking time from checkpoint readiness to the post-reranking CUDA
snapshot fell from approximately 34.4 seconds at batch 128 to 29.0 seconds at
batch 256 (about 15.7% faster). The complete selected stage fell from 279.48 to
275.36 seconds. Post-reranking process memory increased from 4,306/5,988 MiB to
6,460/9,454 MiB on GPU 0/GPU 1, leaving 8,449/5,455 MiB free. All reported
fixed-k and threshold metrics, including the selected threshold 5.746094, were
identical at the displayed precision. This establishes metric parity for the
smoke subset, not yet exact per-pair score or ranking parity; that stronger claim
requires direct comparison of the two run artifacts.
The active multi-dense notebook now promotes this measured treatment to a full
438-query validation: it writes a distinct run identity containing
`packed_stream_b256`, so no earlier dev, OOF LightGBM, or submission artifact
can be silently reused. Its comparison table reports deltas against the
documented 2026-10-27 multi-dense result (`0.663198` pooled score-gap F2 and
`0.660836` nested-OOF score-gap F2), while preserving the earlier result as a
reference rather than overwriting it. The owner subsequently reported that the
full batch-256 run completed. All reported epoch-3 fixed-k and threshold metrics
matched the prior full run at six decimal places, including
threshold-fallback-2 F2 `0.594003` and fixed-top-2 F2 `0.529209`. LightGBM was
close but not identical: pooled score-gap F2 was `0.660765` versus `0.663198`
(`-0.002433`), nested-OOF score-gap F2 was `0.657352` versus `0.660836`
(`-0.003484`), threshold-fallback-2 F2 was `0.648366` versus `0.647531`
(`+0.000835`), and fixed-top-2 F2 was `0.568361` versus `0.574888`
(`-0.006527`). These owner-reported values verify reranker metric parity but not
LightGBM parity. Exact artifact comparison is still required to distinguish
small FP16 score changes from LightGBM training or feature-construction
variation.
The historical reference was traced to Git commit `39a6232` rather than inferred
from the later conservative config. At that revision the completed treatment
used `global_stream`, neural batch 128, outer pair batch 1,024, the same epoch-3
checkpoint, retrieved-chunk source, top-50 candidate pool, 512 pair tokens,
Khoản–Điểm 448/64 representations, two CUDA devices with FP16, and five-fold
LightGBM with seed 42. Therefore the new comparison changes both batching mode
(`global_stream` to `packed_stream`) and neural batch size (128 to 256); it is
not a comparison against `per_query` batch 8.
Candidate sets, token budgets, model weights, and scoring remain unchanged.
The 1,024-token article-rechunk controls retain their prior safe batches because
the benchmark does not cover their activation footprint.
Metadata records both settings plus total scored pairs and
scorer-call count. Unit tests verify identical article rankings and scores for
both article-rechunk and direct retrieved-chunk paths. The following isolated
benchmark measures throughput and memory; it does not replace end-to-end
quality evaluation.

The reranker-only two-T4 benchmark was executed on 2026-10-27 with 500 real
legal pairs arranged as ten question-length-stratified query groups of 50
candidates. Median throughput over seven repeats was approximately 29.03,
78.59, and 89.37 pairs/second for neural batch sizes 8, 64, and 128,
respectively. Thus batch 64 was about 2.71 times batch 8, while batch 128 was
about 3.08 times batch 8 and 1.14 times batch 64. Peak allocated memory on the
most-used GPU was 3,156 MiB, 3,210 MiB, and 3,659 MiB. Against batch 8, both
larger batches retained 100% sampled top-1 agreement, 100% mean top-3 overlap,
and mean rank correlation 0.999962. FP16 logits were not bit-identical: both
comparisons had maximum absolute difference 0.045898, mean absolute difference
about 0.0021, and 95th-percentile difference 0.011719. These are measured
engineering results, not retrieval-quality results. Batch 128 is the benchmark
winner, but a full 438-query run remains the final end-to-end confirmation.

Every current config evaluates fixed final cutoffs 1 through 5 from the same
saved ranking and writes `metrics_by_output_k.json`; this does not rerun
retrieval or reranking. Reranker configs additionally perform an exact sweep
over their observed logits for global `score >= threshold` policies with
empty-result-only top-1 and top-2 fallbacks. The best policy for each fallback,
including its output-size distribution, is written to
`metrics_by_score_threshold.json`. These thresholds are tuned diagnostics on
the development split, are specific to the checkpoint's uncalibrated score
scale, and must not be reported as an unbiased result on those same queries.
The fixed `retrieval.output_k` remains the primary `metrics.json` policy for
backward comparability.

The project owner reported the completed Kaggle run on 2026-09-30 with
Precision `0.2549467275`, Recall `0.6392313546`, and F2 `0.4911637969` on the
same 438-query seed-42 development split. Its candidate Recall@1/5/10/20/50/100
was `0.4010654490`, `0.6918569254`, `0.7748314851`, `0.8376168732`,
`0.8941128506`, and `0.9248804088`, exactly matching the frozen first stage as
intended. Relative to that first stage, zero-shot reranking improves Precision
by `0.003805`, Recall by `0.013470`, and F2 by `0.009191` absolute (about 1.91%
relative). This is a real but modest gain for the additional inference cost.
The run remains user-reported pending inspection of its saved config,
predictions, split IDs, and run metadata.

The project owner subsequently supplied the final scoreboard from
`notebooks/archive/kaggle_overnight_experiments.ipynb` on 2026-09-30. The following compact view
records the strongest policy for each principal reranker treatment; values are
user-reported pending inspection of the archived run artifacts:

| Treatment | Policy | Precision | Recall | F2 |
|---|---|---:|---:|---:|
| Retrieved chunks 50 | threshold + empty-result top-2 fallback | 0.348176 | 0.646271 | 0.551787 |
| Điểm 512, reranker pair/document 1024/512 | threshold + empty-result top-2 fallback | 0.345146 | 0.645700 | 0.549925 |
| Điểm fallback 448, reranker pair/document 512/448 | threshold + empty-result top-2 fallback | 0.402820 | 0.604338 | 0.549371 |
| Điểm 512, reranker pair/document 1024/768 | threshold + empty-result top-2 fallback | 0.343677 | 0.643798 | 0.548075 |
| Điểm fallback 448, reranker pair/document 512/448 | fixed top-2 | 0.336758 | 0.571233 | 0.501409 |
| Retrieved chunks 50 | fixed top-2 | 0.336758 | 0.570548 | 0.500987 |
| Điểm 512, reranker pair/document 1024/512 | fixed top-2 | 0.335616 | 0.569406 | 0.499778 |
| Điểm 512, reranker pair/document 1024/768 | fixed top-2 | 0.333333 | 0.567047 | 0.497310 |

For the retrieved-chunks treatment, the supplied best fallback-2 threshold is
`score >= 4.015625`; it gives mean output size `3.502283`, median `2`, minimum
`1`, and maximum `31`. Its fallback-1 threshold is `3.826171875`. For the
448-token efficiency treatment, the supplied best fallback-2 threshold is
`5.03515625`, while its fallback-1 threshold is `3.912109375`. These thresholds
are configuration-specific cross-encoder logits and must not be transferred
between the two treatments.

Among fixed policies, top-2 is consistently better than top-3 for all four
reranker treatments. The best fixed result is the 448-token efficiency
treatment at F2 `0.501409`, only `0.000422` above direct retrieved-chunk
reranking; this gap is too small to support a strong quality claim without
paired analysis. Direct retrieved-chunk reranking has the best dev-tuned
threshold result at F2 `0.551787`. That threshold and fallback were selected on
the same development queries used to report the score, so `0.551787` is a
policy-selection diagnostic, not an unbiased test estimate. The fixed top-2
policy is the safer submission policy unless a threshold was frozen before
evaluation on a separate held-out split. The optional token-window reranker
control was absent from the supplied scoreboard and therefore has no recorded
result here.

The measured development F2 must not be compared as if it were evaluated on
the same examples as the organizer two-stage baseline's published private-test
F2 (`0.5512` in the conference paper and `0.5515` in the extended paper).
There are also material method differences: the organizer trained its
cross-encoder with positives and BM25 hard negatives and selected a final
top-two output, whereas this run uses an unfine-tuned public reranker and a
fixed top-three output. The implemented multi-cutoff and threshold analyses now
provide the required low-cost output-policy comparison; paired per-query
first-stage-versus-reranker error analysis remains pending. A reranker trained
with leakage-safe hard negatives is the scientifically justified next modeling
step if the zero-shot result remains weak; changing the candidate pool cannot
recover this run's ranking mistakes among candidates already present.

### Retrieved-evidence reranking

The controlled `reranking.candidate_source=retrieved_chunks` treatment keeps
chunk identity through the second stage. Dense retrieval first produces
`retrieved_chunk_depth` ranked representations without collapsing duplicate
`aid` values. Each chunk receives weighted RRF from its dense chunk rank plus
the BM25 rank of its parent article when present. The best
`candidate_pool_size` fused chunks go directly to the cross-encoder, whose
scores are then max-aggregated by `aid`. With the default pool, this scores
exactly 50 query–chunk pairs per query and never reopens the full candidate
articles to score unrelated chunks.

This changes the candidate unit and is therefore a treatment, not merely a
runtime rewrite of article-level WRRF. Multiple selected chunks can belong to
one article, so 50 chunks may yield fewer than 50 unique articles. In this mode
Recall@K describes the unique article order induced by the selected chunks;
the original article-level hybrid list remains in
`first_stage_predictions.jsonl`. `reranker_chunk_candidates.jsonl` records the
selected chunk IDs, parent article IDs, types, and fusion scores. The default
remains `article_rechunk` for backward comparability.

### First reranker fine-tuning protocol

The first supervised reranker trial freezes the selected Điểm 448/64 dense
representation, WRRF `rrf_k=20`, sparse/dense weights `0.2/0.8`, top-50 hybrid
candidate pool, and reranker pair/document budgets 512/448. The deterministic
seed-42 split assigns 1,752 queries to training and excludes all 438 development
queries from mining and optimization.

For each train query, gold aids are positives and every gold aid is excluded
from the negative pool. The zero-shot `BAAI/bge-reranker-v2-m3` scores all Điểm
representations for each gold aid and top-50 non-gold candidate aid; max-score
selection chooses one weakly supervised passage per aid. The five highest
scoring non-gold passages become hard negatives. This matches inference-time
max aggregation but inherits the zero-shot model's within-article passage
selection bias; the emitted audit JSONL preserves qid, aid, chunk ID, and score
for later inspection.

To prevent pathological appendix articles from creating unbounded query-passage
pairs during mining, at most eight representations per negative aid and 32 per
gold aid are passed to the zero-shot teacher. The larger gold cap reduces the
risk of discarding a semantically relevant positive passage; gold aids are few,
so it adds little cost. Above either cap, a deterministic lexical-overlap
prefilter retains the most query-matching representations, with chunk order as
the stable tie-break. Metadata records both caps, how many gold and negative
aid instances were filtered, representation counts before/after filtering,
total scored pairs, and the maximum pairs for any query. This prefilter affects
training-data mining only; evaluation follows each candidate flow exactly.

The initial training treatment uses the official FlagEmbedding encoder-only
reranker trainer, group size six (one sampled positive plus five negatives),
learning rate `2e-5`, two epochs, FP16, gradient checkpointing, and two-GPU DDP.
Checkpoint selection and a new threshold sweep use only the fixed development
split. Zero-shot thresholds are invalid after fine-tuning because the logit
scale changes.

On 2026-10-01, the project owner reported that an earlier Kaggle execution of
the Khoản–Điểm dual-candidate notebook completed both training epochs and saved
the final model plus `checkpoint-110` and `checkpoint-220` under
`/kaggle/working/khoan_diem_finetune_dual/bge_reranker_v2_m3_khoan_diem_448`.
The final-model message appeared after approximately 22,071 seconds (about 6.1
hours). A PyTorch warning that `destroy_process_group()` was not called occurred
after distributed training; because the final model and both epoch checkpoints
were already present, this is currently treated as a non-fatal DDP cleanup
warning rather than evidence of a damaged checkpoint. The notebook subsequently
loaded that checkpoint and entered the 438-query development reranking stage.

This completed training run is user-reported and its development metrics are
not yet independently verified from copied artifacts. Its mining variant also remains unresolved until the saved
`reranker_khoan_diem_train_448_metadata.json` is inspected: absence of cap keys
identifies the original unbounded miner, a negative cap without a distinct
positive cap identifies the intermediate 8/8 miner, and values 8/32 identify
the current miner. A second run using the current code was still executing when
this note was recorded. Preserve both checkpoints and compare their fixed-k and
newly tuned threshold metrics on the identical persisted seed-42 development
split before selecting one; do not infer quality from runtime or successful
checkpoint creation alone.

The same earlier run subsequently completed its `article_rechunk` development
evaluation. The project owner reported fixed top-1/2/3/4/5 F2 values of
`0.313568`, `0.327196`, `0.330343`, `0.315058`, and `0.307827`. Its primary
top-2 result was Precision `0.219178`, Recall `0.373174`, and F2 `0.327196` on
438 queries. Dev-tuned threshold policies reached F2 `0.378851` at threshold
`4.339844` with fallback 1, and F2 `0.383690` at threshold `4.117188` with
fallback 2. These values are dramatically below the matched zero-shot
Khoản–Điểm 512/448 article-first results (top-2 F2 `0.505002`, threshold plus
fallback-2 F2 `0.555900`). Therefore this fine-tuned checkpoint must not replace
the zero-shot reranker or be used for submission. Its direct-chunk result may
still be retained as a diagnostic, but cannot rescue the failed article-first
training treatment. Before another full fine-tuning run, inspect its mining
metadata, audit samples, trainer loss log, and final/checkpoint-110 development
scores to distinguish passage-label noise, optimization collapse, and
overfitting. The still-running current-cap replication is not presumed better
until measured.

The same checkpoint also completed the `retrieved_chunks` branch before the
Kaggle job's later archive failure. The project owner reported top-1/2/3/4/5 F2
of `0.461639`, `0.488633`, `0.471312`, `0.442729`, and `0.416635`; top-2 had
Precision `0.329909` and Recall `0.555441`. Dev-tuned fallback-1 reached F2
`0.532117` at threshold `2.343750` with mean output `3.180`, while fallback-2
reached F2 `0.547714` at threshold `3.066406` with mean output `2.635`. This is
far stronger than the same checkpoint's article-rechunk result, supporting a
representation-alignment diagnosis: training selected and optimized chunk-like
Khoản–Điểm passages, and direct retrieved-chunk inference supplies the same
kind of input, whereas article-first inference reopens many representations
inside each candidate article. It still does not establish an improvement over
the matched zero-shot direct-chunk control and therefore does not reverse the
decision to reject this fine-tuned checkpoint for submission.

A later user-reported execution of the repaired Khoản–Điểm notebook produced a
stronger direct retrieved-chunk result on the same 438-query development split:
top-1/2/3/4/5 F2 `0.456765`, `0.511099`, `0.486778`, `0.457327`, and `0.432133`.
Its threshold policies reached F2 `0.560273` at threshold `3.744141` with
fallback 1 and F2 `0.574508` at threshold `3.722656` with fallback 2. The latter
exceeds the matched zero-shot direct-chunk development F2 `0.555678` by
`0.018830`. These values are user-reported development measurements, not an
unbiased test estimate. Preserve the exact mining metadata and checkpoint
identity before attributing the gain to a particular training treatment.

The Kaggle run completed both evaluations before failing while archiving all of
`/kaggle/working/khoan_diem_finetune_dual`. The partial gzip file reached about
3.6 GB and exhausted the filesystem; papermill then could not save the executed
notebook and the local `__notebook__.ipynb` became truncated. These are artifact
packaging failures, not model-training or metric-computation failures. The four
reported leaked semaphores are a non-fatal multiprocessing shutdown warning.
Future packaging must exclude embedding caches and duplicate epoch checkpoints,
write a small manifest/results archive separately, and package only explicitly
selected model weights when storage capacity has been checked.

The archived dual-candidate notebook now implements that policy: its final ZIP
contains metrics, predictions, mining metadata/audit, scoreboard, and submission
ZIPs only. It excludes model directories, FlagEmbedding data cache, optimization
JSONL, and dense embedding caches. The former whole-WORK `tar` command was the
direct cause of the terminal disk failure because it wrote a second compressed
copy while all large source artifacts were still present.

Based on these comparisons, the selected main pipeline is now hybrid
Khoản–Điểm 448/64, WRRF, dense chunk depth 100, direct reranking of the top 50
fused chunks with pair maximum 512, and max aggregation by `aid`.
`configs/hybrid_reranker_khoan_diem.yaml` records those defaults.
Article-rechunk remains a candidate-flow ablation. This selects the inference
architecture, not the failed fine-tuned weights: zero-shot remains the main
model until an aligned checkpoint beats its matched zero-shot control.

The exact user-reported zero-shot development baseline for this selected case,
`hybrid_bge_m3_khoan_diem448_wrrf20_rerank_retrieved_chunks50_512_seed42_multik`,
has top-1/2/3/4/5 F2 `0.474457`, `0.505002`, `0.495397`, `0.470399`, and
`0.441725`. Dev-tuned fallback-1 reaches F2 `0.545228` at threshold `3.912109`
with mean output `3.737`; fallback-2 reaches F2 `0.555678` at threshold
`4.234375` with mean output `3.495`. These are the matched targets for the new
aligned fine-tuning notebook. The threshold result is a development-selected
diagnostic and must not be described as an unbiased test estimate.

### Aligned direct-chunk BCE: two-epoch result and handoff

On 2026-10-02, the project owner reported completion of the aligned
direct-retrieved-chunk pointwise-BCE run from
`notebooks/archive/kaggle_finetune_main_retrieved_chunks_bce.ipynb`. The selected
epoch-2 snapshot was `checkpoint-2484`. It used the unchanged Khoản–Điểm
448/64 hybrid first stage, direct reranking of 50 fused chunks at pair length
512, and max aggregation by `aid`. The matched 438-query development results
were:

| Model | Policy | Threshold | Precision | Recall | F2 |
|---|---|---:|---:|---:|---:|
| Zero-shot | threshold + fallback 2 | 4.234375 | 0.358609 | 0.644178 | 0.555678 |
| Epoch 1 | threshold + fallback 2 | -0.178589 | 0.467833 | 0.632686 | 0.591033 |
| Epoch 2 | threshold + fallback 2 | 1.576172 | 0.449348 | 0.650381 | **0.596965** |

Epoch 2 therefore improved over its matched zero-shot development control by
`0.041287` F2 absolute (about `7.43%` relative) and is the selected two-epoch
checkpoint. Its frozen development policy is score `>= 1.576171875` with an
empty-result top-2 fallback. This satisfies the predeclared requirement that an
aligned fine-tuned checkpoint beat the matched zero-shot control before test
inference.

The owner also reported private-leaderboard sensitivity submissions for epoch
2. These are post-hoc leaderboard probes, not unbiased test estimates:

| Threshold | Private F2 |
|---:|---:|
| 1.0 | 0.5385 |
| 1.25 | 0.5415 |
| 1.4 | 0.5405 |
| 1.5 | 0.5391 |
| 1.65 | 0.5394 |
| 1.75 | 0.5381 |
| 2.0 | 0.5407 |
| 2.5 | 0.5417 |
| 3.0 | **0.5422** |

A prior zero-shot private submission at threshold `4.1` with fallback 2 scored
`0.5316`; a prior zero-shot public submission at threshold `4.0` scored
`0.5319`. Comparing the private values gives an observed gain of `0.0106` F2
for the leaderboard-selected epoch-2 threshold 3.0, but this is not a clean
model-only estimate because the threshold was selected after observing private
scores. The private epoch-2 sweep spans only `0.0041` F2 and is non-monotonic,
so it does not establish a sharply optimal transferable threshold. For research
reporting, retain `1.576171875` as the development-selected primary policy and
label threshold 3.0 as private-leaderboard-tuned sensitivity analysis.

The subsequent four-epoch run from
`notebooks/archive/kaggle_finetune_main_retrieved_chunks_bce_epochs3_4.ipynb` retained
and evaluated epochs 3 and 4. The project owner reported the following matched
development results on 2026-10-03:

| Model | Policy | Threshold | Precision | Recall | F2 |
|---|---|---:|---:|---:|---:|
| Epoch 3 | threshold + fallback 2 | 5.199219 | 0.487123 | 0.648288 | **0.608053** |
| Epoch 4 | threshold + fallback 2 | 5.582031 | 0.475216 | 0.650951 | 0.606122 |

Epoch 3 is selected by development F2. Epoch 4 is not uniformly worse—it
improves fixed top-2 from `0.541029` to `0.559523`—so the result supports a
threshold-policy plateau after epoch 3 rather than a broad claim of neural
overfitting. Because the four-epoch run has a different total linear-scheduler
horizon, its intermediate checkpoints are not a single directly comparable
learning curve with the earlier two-epoch run.

The frozen epoch-3 development policy, score `>= 5.19921875` with fallback 2,
subsequently scored private F2 `0.5589`. Neighboring finished private probes
scored `0.5579` at threshold `5.1` and `0.5552` at `5.3`, so the preselected
threshold transferred near the observed local optimum. A public submission at
the frozen threshold displayed F2 `0.5811` while still marked `Running`; do not
record that public value as final until Codabench reports `Finished`. All
leaderboard values remain user-reported pending artifact verification.

Aligned fine-tuning uses the exact fused retrieved chunks as negatives. The
zero-shot reranker orders them; mining retains a top-30 non-gold pool, discards
the top five suspicious hits, and samples ten semi-hard chunks deterministically.
Because labels identify articles rather than passages, positives remain weakly
supervised: the teacher selects the maximum-scoring same-448/64 representation
inside each gold article and the audit preserves every selection. This is
FPT-inspired but not an exact FPT reproduction. The full-article FPT/BCE
notebook remains a separate published-method control.

External-method review after this failure found that the strongest documented
DRiLL hard-negative recipe is deliberately less aggressive: use gold articles
directly as positives, consider non-gold candidates within the top 30, discard
the top five suspicious near-positive hits, and randomly sample from the
remainder. By contrast, the failed run used a zero-shot teacher to choose one
positive chunk inside each gold article and then chose the five highest-scoring
non-gold candidates as negatives. This makes positive passage-label noise and
false-negative contamination the primary hypotheses; the 8/32 representation
cap is secondary until the audit metadata shows otherwise. The published FPT
system also used pointwise BCE, while this run used FlagEmbedding's grouped
training objective, so loss family is a separate ablation rather than an
assumed implementation detail.

Before another full GPU run, follow this order: (1) evaluate the saved
`checkpoint-110`; (2) inspect mining metadata, trainer loss, and a deterministic
sample of positive/negative audit records; (3) implement an article-level,
false-negative-aware mining control without structural chunk supervision; (4)
run a small deterministic canary to verify that the checkpoint improves rather
than collapses ranking; (5) run one full controlled training treatment; and
only then (6) reintroduce Khoản–Điểm positive representations as an independent
structure treatment. Do not simultaneously change mining policy, objective,
length budget, and legal representation in the first recovery experiment.

The FPT-style recovery treatment is now implemented separately rather than
mutating the failed protocol. `configs/hybrid_reranker_fpt_article.yaml` keeps
the selected Khoản–Điểm 448/64 hybrid first stage, but reranks top-100 article
candidates as complete articles with a 1,024-token pair budget. Training-data
preparation uses only the deterministic 1,752-query train side. It excludes
gold or candidate articles above 1,024 reranker tokens, uses every eligible
gold article directly as a positive, forms a top-30 eligible non-gold pool from
the zero-shot article reranker, discards the top five hits, and deterministically
samples 10 negatives when more than 15 remain or five otherwise. This is the
published FPT.AI sampling rule; the seed and selected aids are persisted.

The emitted optimization JSONL contains exactly `query`, `passage`, and binary
`label` fields. A separate audit JSONL retains qid, positive/negative aid, and
token lengths. `scripts/train_cross_encoder_bce.py` trains the one-logit Hugging
Face sequence classifier with pointwise binary cross-entropy, two epochs,
effective two-GPU batch size 16, learning rate `2e-5`, and epoch checkpoints.
The matched zero-shot model, epoch 1, and epoch 2 must all be
evaluated with the same article representation, first-stage candidates, dev
qids, and output-policy sweep. The best fine-tuned development F2 must exceed
the matched zero-shot result before test inference is justified. This treatment
changes supervision unit, negative policy, objective, and length budget as a
faithful published-method reproduction; it must not be interpreted as an
isolated single-factor ablation. A later structure experiment can reintroduce
Khoản–Điểm reranker representations after this article-level control succeeds.

The first reported execution of the aligned direct-chunk BCE treatment failed
before its first optimizer step on 2026-10-01. With the installed
SentenceTransformers/Transformers versions, `CrossEncoderTrainer` replaced the
loss object's CrossEncoder reference with `DistributedDataParallel`; then
`BinaryCrossEntropyLoss` called the CrossEncoder-only `preprocess` method on
that wrapper and raised `AttributeError`. The completed train-only mining JSONL
is unaffected. Training now uses a small explicit PyTorch DDP loop with the
same tokenizer, one-logit model, unweighted pointwise BCE, data, sequence
budget, effective batch size, optimizer settings, seed, and linear warmup.
This is an execution-engine repair, but it may still cause numerical differences
from the former Trainer and is recorded in `training_provenance.json`. The
two-rank, one-optimizer-step smoke path writes provenance without model weights,
but the selected notebooks now skip this already-validated smoke by default:
it starts a separate DDP process and does not validate the subsequent fresh
full-training process. It remains an explicit debug flag. The trainer also
destroys the process group on exit and expresses the warmup ratio as an explicit
number of scheduler steps.

For the replacement BCE trainer, the root final model is hard-linked from the
last epoch checkpoint on filesystems that support hard links, with symbolic
links as the fallback. Every completed epoch remains a loadable Hugging Face
snapshot. Only the newest checkpoint retains optimizer, scheduler, scaler, and
per-rank RNG state, bounding resume-state storage while preserving every epoch
for model selection. Exact epoch-boundary resume validates training data and
hyperparameters and requires a scheduler horizon declared before the original
run; otherwise loading an earlier model is explicitly a warm start. Before
optimization, the trainer estimates all epoch model snapshots plus one Adam
state and requires an additional 2 GiB reserve. Artifact ZIPs exclude model
snapshots and embedding caches. Before writing a new epoch's resume state, the
trainer removes the previous checkpoint's optimizer/RNG state while retaining
its model snapshot. This avoids a transient second Adam state that could exceed
the 20 GiB Kaggle quota; a crash in that short replacement interval preserves
all epoch models but may forfeit exact optimizer-state resume until the new
state finishes writing.

### Hierarchical Khoản–Điểm representation

`khoan_diem` is an independently selectable dense and reranker representation.
It keeps every parsed Khoản that fits the configured token ceiling as one
representation, even when the Khoản contains Điểm. An oversized Khoản with a
valid Điểm sequence greedily packs consecutive complete Điểm under the same
ceiling while repeating the Khoản lead-in. Only an individually oversized Điểm
uses overlapping token windows after subtracting lead-in, prefix, and special-
token cost. Ambiguous or absent Điểm structure, and lead-ins that exhaust the
usable budget, fall back to complete-Khoản token windows. Articles without a
safe Khoản parse retain the existing complete-article token-window fallback.

The method never treats 512 as a fill target for otherwise short Khoản and does
not merge separate Khoản. This keeps the first comparison attributable to a
legal boundary strategy; cross-Khoản packing remains a future independent
ablation. `configs/hybrid_reranker_khoan_diem.yaml` and the dedicated Kaggle
notebook freeze this representation without changing the earlier Điểm suite.

The project owner reported the completed
`notebooks/archive/kaggle_khoan_diem_experiments.ipynb` scoreboard on 2026-10-01. These values use
the same 438-query seed-42 development split and remain user-reported pending
inspection of the archived configs, predictions, candidate rankings, and run
metadata.

At the first stage, hierarchical Khoản–Điểm did not improve over the strongest
non-structural control. BGE-M3 Khoản–Điểm 512/64 obtained fixed top-2/top-3 F2
of `0.460912`/`0.460627`, versus `0.483955`/`0.475098` for BGE-M3 token windows.
The Khoản–Điểm hybrid reached top-3 F2 `0.474606`, below the earlier Điểm hybrid
F2 `0.481973`. Thus this representation is not supported as a stronger
first-stage retriever by the current development results.

After reranking, however, every matched end-to-end Khoản–Điểm treatment
outperformed its earlier Điểm counterpart:

| Treatment | Điểm top-2 F2 | Khoản–Điểm top-2 F2 | Difference | Điểm threshold/fallback-2 F2 | Khoản–Điểm threshold/fallback-2 F2 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Pair/document 1024/768 | 0.497310 | 0.503088 | +0.005778 | 0.548075 | 0.558422 | +0.010347 |
| Pair/document 1024/512 | 0.499778 | 0.503875 | +0.004097 | 0.549925 | 0.557295 | +0.007370 |
| Pair/document 512/448 | 0.501409 | **0.505002** | +0.003593 | 0.549371 | 0.555900 | +0.006529 |
| Retrieved chunks 50 | 0.500987 | 0.503875 | +0.002888 | **0.551787** | 0.557849 | +0.006062 |

The best fixed Khoản–Điểm result is the efficient 512/448 treatment at F2
`0.505002`; the best dev-tuned variable-output result is the 1024/768 treatment
at F2 `0.558422`. The fixed improvements are small (`+0.002888` to `+0.005778`)
but directionally consistent across all four treatments. Threshold improvements
are larger (`+0.006062` to `+0.010347`) but retain the same optimistic bias from
selecting and evaluating a threshold on one development split.

These are end-to-end comparisons: changing from Điểm to Khoản–Điểm changes the
dense candidates as well as the reranker representations. The table therefore
does not isolate a reranker-only structure effect. Candidate Recall@k, paired
per-query predictions, and the optional fixed-first-stage reranker-structure
control must be inspected before attributing the gain specifically to richer
reranker context. The defensible current conclusion is narrower: hierarchical
Khoản–Điểm hurts standalone first-stage ranking in this setup, yet produces a
small, consistent improvement in the complete reranked pipeline.

One additional controlled efficiency treatment combines dense Khoản–Điểm
448/64 and pair max 512 from the best fixed article-first case with direct
top-50 retrieved-chunk reranking. It keeps WRRF20, weights, dense chunk depth
100, checkpoint, split, and final policies fixed. The comparison changes only
whether top-50 selection occurs after article aggregation or at the fused chunk
level. Its result is pending execution; report both quality and the exact
`retrieved_chunk_pairs`/unique-aid coverage rather than assuming that 50 chunks
represent 50 articles.

### Query-level retrieval-deficit audit

`scripts/analyze_query_retrieval.py` explains first-stage Recall@50 from a
completed development run. It discovers component names from
`component_rankings.jsonl`, so adding, removing, or replacing dense models does
not change the analysis. For every query and gold `aid`, it records ranks in
each component, fused article retrieval, the unique-article candidate list
actually passed to reranking, epoch-3, and optional later rankings such as
LightGBM OOF.

The per-query contribution is `(1 - query Recall@50) / query_count`; therefore
the contributions sum exactly to `1 - macro Recall@50`. Gold attribution
separates items missed by all components, items found by a component but fused
outside top 50, and items present before reranking but demoted later. The union
of every component's top 50 is an oracle with a budget potentially greater than
50. Component uniqueness and sparse-rescue counts are descriptive, not causal
leave-one-out estimates; causal attribution requires re-fusion after removing
one component with all other settings fixed. Direct chunk candidates are a
separate stage because 50 chunks can represent fewer than 50 unique articles.

### Planned Harrier + BGE-M3 multi-dense control

`notebooks/archive/kaggle_multi_dense_harrier_bge.ipynb` is the controlled dense-peer
replacement experiment for the current multi-dense pipeline. It replaces
multilingual-E5 with `BAAI/bge-m3`, while retaining BM25S with VnCoreNLP,
VietLegal-Harrier, the persisted seed-42 split, `top_chunks`, the epoch-3
cross-encoder, packed-stream reranking with neural batch 256, and the existing
OOF/final LightGBM stages. Same-law neighbor expansion and explicit-reference
expansion are both disabled. Thus this notebook does not test legal structure;
it asks only whether BGE-M3 is a better complementary dense peer for Harrier.

The development screen crosses WRRF `k` in `{10, 20, 40, 60}` with ten explicit
BM25/Harrier/BGE weight triples. Selection is lexicographic by candidate
Recall@50, then Recall@20, then fixed top-3 F2, with declaration order as the
last deterministic tie-break. This is development tuning and must not be
reported as an untouched evaluation. The selected treatment must be compared
with Harrier + multilingual-E5 using first-stage recall, paired query-level
wins/losses, reranker rankings, nested-OOF LightGBM, and final predictions;
matching scores are not assumed because the dense component intentionally
changes. No measured result is recorded until the notebook completes and its
artifacts are inspected.

Two matched follow-ups complete the three-pair comparison. The
`kaggle_multi_dense_harrier_vietlegal_e5.ipynb` treatment pairs Harrier with
`mainguyen9/vietlegal-e5`; the
`kaggle_multi_dense_harrier_e5_retuned.ipynb` treatment reruns Harrier with
`intfloat/multilingual-e5-large` using the expanded WRRF screen. Both disable
same-law neighbor and explicit-reference expansion and retain the BGE-M3-
tokenized Khoản–Điểm chunk store, reranker, split, selection policy, and
LightGBM stages. The three no-expansion notebooks therefore differ only in the
second dense peer, apart from model-specific query/document prefixes and
embedding caches.

The retuned Harrier + multilingual-E5 notebook is also the intended
no-reference control for the reference-enabled main notebook. That comparison
is valid only if the resolved configs confirm identical split IDs, WRRF search
space and selected weights, candidate selection, reranking, and downstream
policy. If reference expansion changes the WRRF winner, report both the full
end-to-end treatment effect and a fixed-WRRF reference-only ablation; do not
attribute the combined difference solely to graph expansion.

### Pointwise MIL checkpoint evaluation boundary

The completed pool-50 and pool-100 pointwise MIL runs are evaluated with the
matching frozen development handoff rather than repeating first-stage
retrieval. The evaluation-only notebooks and
`scripts/evaluate_reranker_checkpoints.py` compare the pinned Vietnamese warm
start with all five adaptation snapshots on identical candidates. Checkpoints
are loaded sequentially in separate subprocesses; no two reranker models are
resident together, and process exit is the cleanup boundary before the next
snapshot. Fixed top-2 is the primary checkpoint comparison. The score-threshold
fallback-2 result is a development-tuned calibration diagnostic and must not be
reported as an untouched test estimate. No checkpoint result is measured until
the corresponding notebook run completes.

Measured checkpoint evaluation completed on 2026-10-30 for both frozen
handoffs. Under the predeclared fixed-top-2 gate, pool 50 epoch 1 is the overall
winner: precision `0.393836`, recall `0.657610`, and F2 `0.579928`. Pool 100's
best fixed-top-2 checkpoint is epoch 2: precision `0.385845`, recall `0.639612`,
and F2 `0.565259`. Thus pool 50 epoch 1 is the reranker finalist for subsequent
LightGBM comparisons. As a non-selection diagnostic, the best tuned
threshold/fallback-2 result is also pool 50 epoch 1 with F2 `0.623556`; pool
100's best is epoch 1 with F2 `0.610202`. Sources are the two archived
`checkpoint_scoreboard.csv` files in
`pointwise_mil_pool{50,100}_checkpoint_eval_metrics.tar.gz`.

### Pool-50 epoch-1 LightGBM feature screen

The post-selection feature screen completed on 2026-10-30 using checkpoint
`1208`, the verified pool-50 handoff, 438 development queries, and 13,348
candidate rows containing 522 positive rows. All four schemas use identical
query-grouped five-fold OOF assignments. The measured results are:

| Schema | Features | Fixed top-2 F2 | Pooled threshold/fallback-2 F2 | Nested-OOF score-gap F2 |
|---|---:|---:|---:|---:|
| `base_v1` | 15 | 0.583262 | 0.664100 | 0.648388 |
| `enhanced_v1` | 41 | 0.594233 | **0.669570** | 0.649468 |
| `enhanced_v2` | 61 | 0.594233 | 0.663271 | **0.658333** |
| `enhanced_v3` | 69 | **0.597850** | 0.662563 | 0.657139 |

The frozen reranker alone scored fixed-top-2 F2 `0.579928` and tuned
threshold/fallback-2 F2 `0.623556`. Thus every LightGBM schema improved the
fixed-top-2 ranking, while the 61-feature schema produced the strongest
leakage-controlled selector estimate available in this screen. Because four
schemas were compared on the same development data, even this nested estimate
inherits method-selection uncertainty and is not an untouched test estimate.
The 69-feature schema improved fixed top-2 by `0.003616` over 61 features but
reduced nested score-gap
F2 by `0.001194`; removing its eight article-static fields exactly recovers the
61-feature result. The current primary choice is therefore `enhanced_v2` for
score-gap selection, with `enhanced_v3` retained as the fixed-top-2 control.

Group removal is exploratory because many groups were screened on the same
development split. The best observed nested score-gap ablation was
`enhanced_v3` without query/chunk lexical features at F2 `0.664216`; the best
61-feature ablation removed retriever-agreement features at F2 `0.661736`.
These should be treated as hypotheses for one confirmatory reduced-schema run,
not as already selected winners. Split/gain importance likewise remains
descriptive. The source archive is
`pool50_epoch1_lightgbm_feature_screen.tar.gz`; its static table contains
59,628 unique article IDs and records corpus and output SHA-256 hashes.

The complete `enhanced_v2` group-removal screen was:

| Removed feature group | Nested-OOF score-gap F2 | Change from full 61 features |
|---|---:|---:|
| None (full `enhanced_v2`) | 0.658333 | 0.000000 |
| Retriever agreement | 0.661736 | +0.003403 |
| Query/chunk lexical | 0.660375 | +0.002043 |
| First-stage components | 0.658642 | +0.000309 |
| Chunk fusion | 0.657927 | -0.000406 |
| Reference | 0.657108 | -0.001225 |
| Per-chunk reranker distribution | 0.655033 | -0.003299 |
| Query shape | 0.653771 | -0.004561 |
| Chunk length/coverage | 0.651151 | -0.007181 |
| Within-query calibration | 0.640962 | -0.017371 |

Positive changes are observed development differences, not confidence-tested
improvements. The archive retains the corresponding predictions and the full
ablation output for the other schemas.

For `enhanced_v2`, normalized gain importance is dominated by within-query
calibration (`64.00%`), followed by article-level reranker features (`8.95%`),
first-stage components (`7.60%`), chunk fusion (`6.08%`), chunk length/coverage
(`3.91%`), and per-chunk reranker distribution (`3.77%`). The largest individual
features are reranker min-max normalization (`40.01%`), reranker z-score
(`17.20%`), and margin from the reranker top score (`6.52%`). Importance and
ablation do not always agree: first-stage and retriever-agreement groups carry
nonzero gain but their removal slightly improves nested score-gap F2, whereas
the low-gain query-shape group reduces nested F2 when removed. This confirms
that pruning decisions must use held-out group ablation rather than a raw gain
cutoff.

### Experiment ledger: current pointwise finalist

This ledger freezes the executed sequence as of 2026-10-30. Values in the
`Measured outcome` column come from retained artifacts; rows marked exploratory
must not be presented as confirmatory findings.

| Stage | Frozen treatment | Artifact or checkpoints | Status | Measured outcome / decision |
|---|---|---|---|---|
| Pointwise MIL adaptation, pool 50 | Reference on; WRRF `k=10`; BM25/Harrier/E5 `0.15/0.55/0.30`; Vietnamese reranker warm start; five epochs; deterministic seeded-cycle sampling of at most four article chunks per epoch | `1208, 2418, 3630, 4842, 6054` | Completed | All five checkpoints and final model were saved. Training loss is an optimization diagnostic only; there is no separate pair-level `eval_loss`. |
| Pointwise MIL adaptation, pool 100 | Same treatment, changing only the candidate budget and artifact identity | `1238, 2479, 3721, 4961, 6203` | Completed | All five checkpoints and final model were saved. This is the candidate-budget ceiling ablation. |
| Matched checkpoint screen, pool 50 | One checksummed retrieval handoff reused by the warm start and five checkpoints; sequential fresh reranker subprocesses; packed-stream inference batch 256 | `pointwise_mil_pool50_checkpoint_eval_metrics.tar.gz` | Completed | Epoch 1 / checkpoint `1208` won the predeclared fixed-top-2 gate at P/R/F2 `0.393836/0.657610/0.579928`. Its pooled tuned threshold diagnostic was F2 `0.623556`. |
| Matched checkpoint screen, pool 100 | Same protocol with the pool-100 handoff | `pointwise_mil_pool100_checkpoint_eval_metrics.tar.gz` | Completed | Epoch 2 / checkpoint `2479` was the pool-100 fixed-top-2 winner at P/R/F2 `0.385845/0.639612/0.565259`; pool-100 epoch 1 had the best pooled threshold diagnostic at F2 `0.610202`. Pool 100 did not displace pool 50. |
| LightGBM schema screen | Pool-50 checkpoint `1208`; 438 fixed development queries; 13,348 rows, 522 positives; identical five query-grouped folds and seed 42 | `pool50_epoch1_lightgbm_feature_screen.tar.gz` | Completed | `enhanced_v2` (61 features) had the strongest nested-OOF score-gap F2, `0.658333`; `enhanced_v3` (69) had the strongest fixed-top-2 F2, `0.597850`; `enhanced_v1` (41) had the strongest pooled threshold diagnostic, `0.669570`. |
| Feature-group removal and importance | Same saved rows, folds, schemas, and policy search | Ablation and importance files inside the LightGBM archive | Completed, exploratory | Best observed removal was `enhanced_v3` without query/chunk lexical features at nested F2 `0.664216`; best 61-feature removal dropped retriever agreement at `0.661736`. These are hypotheses, not selected winners. |

The LightGBM screen used binary classification with 200 trees, learning rate
`0.03`, 15 leaves, maximum depth 4, minimum child samples 20, full row and
column sampling, L1 `0.1`, L2 `1.0`, and fold-specific deterministic random
states derived from seed 42. `base_v1`, `enhanced_v1`, `enhanced_v2`, and
`enhanced_v3` are nested 15/41/61/69-feature supersets. The static feature
artifact covers 59,628 unique `aid` values and retains its corpus/output hashes.

The selection hierarchy is deliberately asymmetric:

1. Fixed top 2 selected the neural checkpoint and candidate budget, because it
   does not add a learned output-set policy.
2. Nested query-fold OOF score-gap F2 compares learned selectors and is the
   primary generalization estimate for LightGBM.
3. Pooled OOF threshold/fallback-2 and pooled score-gap optima are development
   diagnostics. They may supply frozen deployment parameters only after the
   method is chosen; they are not unbiased estimates of that choice.
4. Gain importance is descriptive. Group-removal ablation is stronger pruning
   evidence, but the many removals in this screen still require confirmation.

Current decision: carry pool-50 epoch 1 / checkpoint `1208` forward. Retain the
full 61-feature `enhanced_v2` schema as the primary LightGBM model and the full
69-feature `enhanced_v3` fixed-top-2 ranking as a control. If compute permits
one more development run, test exactly one predeclared reduced schema informed
by the exploratory ablations; do not continue open-ended single-feature search.

Not yet measured in this branch are an independently confirmed reduced schema,
LightGBM hyperparameter tuning, LambdaRank or another learner, a nested
threshold-policy estimate, uncertainty intervals for the final comparison, and
an official public/private evaluation of the newly fine-tuned finalist. Do not
report any of those as completed or infer leaderboard gains from the 438-query
development results.

The official-inference launcher is now prepared at
`notebooks/kaggle_pool50_epoch1_lightgbm_submission.ipynb`. It freezes
checkpoint `1208`, `enhanced_v2`, seed 42, and score-gap maximum cutoffs
`{3,4,5}`, reproduces the nested-OOF F2 gate before packaging, and emits
public/private score-gap submissions plus fixed-top-2 and pooled-threshold
controls. This records an executable plan, not an executed result: official
scores remain unmeasured until the notebook completes and Codabench returns a
recorded evaluation.

The first official-inference attempt on 2026-10-30 is a failed execution, not a
measured result. Retrieval completed in-process, but residual dense/FAISS GPU
ownership left only 115 MiB free on `cuda:0`; packed reranking at the matched
batch size 256 therefore failed after five of 312 queries. The launcher now
overrides the inherited dev configuration with
`runtime.process_handoff_before_reranking=true`, preserving retrieval artifacts
across an `exec` process replacement before loading the reranker. No official
prediction or score is attributed to the failed attempt.

The repaired launcher additionally treats stage cleanup as part of the runtime
contract. A consumed retrieval handoff is removed only after every split has
been deserialized and split coverage has been verified. Reranker cleanup occurs
only after predictions, chunk features, component rankings, run metadata, and
the submission manifest have been written. These lifecycle changes are
output-invariant; the frozen reranker batch remains 256 and no fallback batch or
model substitution is introduced.

### Repository artifact cleanup

The 2026-10-30 final cleanup leaves only three current notebook entry points:
cache construction, the pool-50 checkpoint-1208 LightGBM feature screen, and
the corresponding official submission launcher. Completed pointwise
mining/training/evaluation notebooks, deferred pairwise notebooks, superseded
reranker treatments, and preparation utilities are retained in
`notebooks/archive/`; they are reproducibility records, not active workflows.

The top level of the ignored `results/` directory retains exactly the three
decision archives used by the current report:
`pointwise_mil_pool50_checkpoint_eval_metrics.tar.gz`,
`pointwise_mil_pool100_checkpoint_eval_metrics.tar.gz`, and
`pool50_epoch1_lightgbm_feature_screen.tar.gz`. Superseded first-stage,
multi-dense, and reranker result bundles were removed from the local ignored
artifact directory after their measured decisions and limitations had been
recorded in the Markdown sources of truth. Reference graph/review artifacts and
parser/design audits remain in their named directories because measured facts
in `docs/dataset.md` and the reference ablation still depend on them.

### Official pool-50 checkpoint-1208 LightGBM results

Codabench returned the following official scores on 2026-10-30 for the frozen
`enhanced_v2` LightGBM submission pipeline. These are measured leaderboard
scores supplied from completed submissions, not development estimates:

| Split | Output policy | Submission ID | F2 |
|---|---|---:|---:|
| Public | Pooled score-gap, maximum cutoffs `{3,4,5}` | 1021732 | 0.6551 |
| Public | Pooled threshold with fallback 2 | 1021667 | 0.6524 |
| Private | Pooled threshold with fallback 2 | 1021640 | 0.6054 |
| Private | Pooled score-gap, maximum cutoffs `{3,4,5}` | 1021639 | 0.6047 |
| Public | Fixed top 2 | 1022898 | 0.5989 |
| Private | Fixed top 2 | 1022897 | 0.5552 |

Score-gap leads threshold by `0.0027` on public, while threshold leads
score-gap by `0.0007` on private. The ordering therefore does not transfer
consistently and the private difference is negligible at the reported
precision. The score-gap public result is close to its pre-recorded nested-OOF
development estimate (`0.658333`, difference `-0.003233`), but its private
score is lower by `0.053633`. Keep score-gap as the predeclared primary policy
and threshold as the documented control; do not switch the selected method
post hoc based on the private leaderboard.

Compared with LightGBM fixed top 2, threshold improves public/private F2 by
`0.0535/0.0502`, and score gap improves it by `0.0562/0.0495`. This isolates
final-set cardinality as the largest measured remaining decision boundary. The
next experiment plan and required report tables are frozen in
`docs/final_report_plan.md`.
