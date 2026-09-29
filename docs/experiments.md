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

Fusion must be explicit. Reciprocal Rank Fusion is suitable for rank-based
fusion. Direct weighted fusion of BM25 scores and cosine similarities requires a
documented normalization strategy.

The dense baseline is decomposed into encoder, cache, and search components in
`configs/dense.yaml`. It uses `intfloat/multilingual-e5-base`, explicit E5 query
and passage prefixes, a recorded maximum sequence length, and exact cosine
search. Exact search is the correctness reference before adding FAISS or another
approximate backend. Corpus embeddings are cached only after validating the
corpus fingerprint, ordered aid mapping, and document-encoder configuration.

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
128-token overlap are an ablation rather than a universal default. For
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

The planned first hybrid method is weighted reciprocal rank fusion (WRRF; also
sometimes shortened to weighted RRF), not direct addition of raw BM25 and cosine
scores. Both retrievers already emit the same `RetrievalResult` structure, so
fusion can consume ranks while preserving each method's original scores for
analysis. Fusion weights and the RRF rank constant must be explicit config
values and tuned only on the development split.

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
  metrics: [precision, recall, f2]
```

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
| Token window 512/64 | 0.207002 | 0.524163 | 0.401217 | 0.325342 | 0.605806 | 0.698407 | **0.796467** | **0.874766** | 0.904659 |
| Recursive token 512/64 | 0.204718 | 0.514650 | 0.395037 | 0.325342 | 0.613416 | 0.705121 | 0.793803 | 0.873625 | **0.905420** |
| Khoản 512/64 fallback | **0.221461** | **0.557154** | **0.427540** | **0.354262** | **0.647663** | **0.728142** | 0.786628 | 0.869684 | 0.901288 |

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
representation while fixed 512/64 remains the appropriate non-structural
control.
