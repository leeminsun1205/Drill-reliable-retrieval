# Local and Kaggle execution

## Intended workflow

Git is the source of truth. The normal workflow is:

1. develop and run lightweight checks locally;
2. commit and push the reproducible code and configuration;
3. clone or update the same repository in Kaggle;
4. attach the dataset through Kaggle inputs;
5. invoke repository scripts rather than reimplementing pipeline logic in the
   notebook;
6. save small metrics and provenance for later review, while keeping large
   embeddings and checkpoints outside Git.

Kaggle notebooks should be thin launchers: install dependencies, resolve input
and output paths, call scripts, and display or export results. Core research
logic belongs in `src/drill_ir/`.

## Kaggle setup

For authenticated Hugging Face downloads, create a Kaggle Secret named
`HF_TOKEN` using a read-only Hugging Face access token, then set it once near
the top of the notebook without printing it or passing it as a CLI argument:

```python
from kaggle_secrets import UserSecretsClient
import os

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Kaggle Secret HF_TOKEN is missing or empty")
os.environ["HF_TOKEN"] = hf_token
```

Child processes launched by later `!python scripts/...` cells inherit this
environment variable. Authentication raises Hub download rate limits and can
make model/tokenizer downloads faster and more reliable; it does not accelerate
GPU encoding or reranking after the files are cached. Never place the token in
the repository, experiment config, command arguments, or notebook output.

After cloning or updating the repository, install it in editable mode from the
repository root. This is the preferred execution path because it makes the
`src/` package importable independently of the notebook working directory:

```bash
%cd /kaggle/working/Drill-reliable-retrieval
!python -m pip install -e .
!drill-validate \
  --corpus /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --queries /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet
```

Dense experiments require the optional model stack:

```bash
!python -m pip install -e '.[dense]'
!drill-run --config configs/dense.yaml \
  --output-root /kaggle/working/results
```

The `dense` extra includes `faiss-cpu`. Active dense configs use
`search.backend: faiss_flat`, `metric: cosine`, and `device: cpu`. This is exact
FAISS Flat search, not ANN. Environments with FAISS GPU bindings may select a
specific `cuda:N`, but CPU is the controlled default to preserve GPU capacity
for encoding and reranking. The legacy `exact` name remains an alias for the
NumPy reference backend `exact_numpy`.

`bm25s` is a core dependency because it is the selected execution backend for
standard BM25. To reproduce its comparison with the native oracle, run
`notebooks/archive/kaggle_bm25_native_vs_bm25s.ipynb`. It builds and verifies separate caches, evaluates both
backends on the same persisted seed-42 split, reports exact ranking overlap at
1/5/10/20/50/100, and emits:

- `/kaggle/working/retrieval_cache_bm25s_vncorenlp.tar`
- `/kaggle/working/bm25_native_vs_bm25s_results.tar.gz`

The notebook marks replacement safe only if all development top-100 rankings
and all metrics match. This gate passed locally on 2026-10-27; rerunning it on
Kaggle verifies portability. A faster runtime by itself is insufficient.

## Retrieval cache preparation

Build expensive retrieval state independently from evaluation. Each invocation
prepares exactly one component and exits, ensuring that multi-GPU workers and
model memory are released before the next dense encoder starts. For the current
multi-dense config, use three separate commands:

```python
!python scripts/prepare_retrieval_cache.py \
  --config configs/hybrid_reranker_multi_dense.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --component sparse \
  --set retrieval.bm25.preprocessing.tokenizer.word_segmenter=vncorenlp \
  --set retrieval.bm25.preprocessing.tokenizer.vncorenlp_jar_path=/kaggle/working/VnCoreNLP/VnCoreNLP-1.2.jar \
  --set retrieval.bm25.cache.directory=/kaggle/working/retrieval-cache/bm25

!python scripts/prepare_retrieval_cache.py \
  --config configs/hybrid_reranker_multi_dense.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --component vietlegal_harrier --devices 2 \
  --set retrieval.dense_profiles.vietlegal_harrier.cache.directory=/kaggle/working/retrieval-cache/vietlegal_harrier_0_6b

!python scripts/prepare_retrieval_cache.py \
  --config configs/hybrid_reranker_multi_dense.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --component multilingual_e5_large --devices 2 \
  --set retrieval.dense_profiles.multilingual_e5_large.cache.directory=/kaggle/working/retrieval-cache/multilingual_e5_large
```

The sparse cache depends on the exact preprocessing configuration, including
the configured VnCoreNLP path. Dense caches depend on model/revision, document
prefix, chunking, normalization, ordered `aid` mapping, and corpus content.
After saving `/kaggle/working/retrieval-cache` as a Kaggle output or dataset,
point the corresponding cache directories at that attached input and enable
`require_hit: true`. Do not copy or commit generated caches to Git.

For parallel Kaggle preparation, use
`notebooks/kaggle_build_retrieval_cache.ipynb`. Clone it into separate Kaggle
notebooks and change only `CACHE_TARGET`: `0` builds VnCoreNLP BM25S, `1`–`7`
select the fixed seven-model dense screening registry, and `8` builds the
shared BGE-M3 Khoản–Điểm 448/64 chunk records without encoding documents. Every notebook builds
one component, then reruns the loader with `--require-existing`; the version is
successful only if the just-written cache passes full validation and reports a
cache hit. The final cell also packages the config, manifest, and cache directory
as one uncompressed `retrieval_cache_<slug>.tar`. Download or upload only this
single archive when moving a cache through Drive or a Kaggle Dataset. The
multi-dense notebook discovers attached archives, extracts them under
`/kaggle/working`, and validates them before use. It searches both
`/kaggle/input` and `/kaggle/working`, so an archive copied from Drive into the
working directory is handled identically. Uncompressed tar is deliberate:
dense float arrays generally gain little from gzip while compression adds a
large CPU and wall-time cost.

For sparse tuning and the matched BM25+ control, run
`notebooks/archive/kaggle_sparse_bm25_plus_screening.ipynb`. It builds and verifies two
VnCoreNLP index caches, evaluates a predefined 12-case BM25 grid
(`k1` in `0.6/0.9/1.2/1.5`, `b` in `0.4/0.6/0.75`) plus the fixed BM25+ control
on the same persisted seed-42 development split, and writes two portable
outputs:

```text
/kaggle/working/retrieval_cache_bm25_plus_vncorenlp_delta1.tar
/kaggle/working/sparse_tuning_bm25_and_bm25_plus.tar.gz
```

The first file is the reusable cache accepted by the multi-dense notebook. The
second contains comparison tables, metrics, configs, split IDs, and run
provenance. The BM25 grid winner is selected by Recall@100, then Recall@50,
then fixed top-3 F2. BM25+ remains fixed at `k1=1.2`, `b=0.75`, and
`delta=1.0`; preprocessing, candidate depth, split, and output cutoffs remain
matched. Treat the winning grid score as development selection, not an unbiased
test estimate.

For user-facing Kaggle notebooks, use the repository scripts consistently. Run
the exact-token audit before selecting a chunk budget:

```python
!python scripts/audit_token_lengths.py \
  --model-name intfloat/multilingual-e5-base \
  --document-prefix "passage: " \
  --query-prefix "query: " \
  --corpus data/legal-corpus.parquet \
  --queries data/train.parquet \
  --thresholds 512,1024 \
  --output /kaggle/working/results/e5_token_lengths.json
```

The article-truncation control remains the default in `configs/dense.yaml`.
Run the fixed tokenizer-window treatment with:

```python
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_e5_token_window_512_64_seed42 \
  --set retrieval.dense.chunking.method=token_window \
  --set retrieval.dense.chunking.max_tokens=512 \
  --set retrieval.dense.chunking.overlap_tokens=64 \
  --output-root /kaggle/working/results
```

Both commands accept explicit Kaggle input paths when the files are not copied
under repository-local `data/`.

After the fixed-window control is measured, run the recursive-separator baseline
with all other settings unchanged:

```python
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_e5_recursive_token_512_64_seed42 \
  --set retrieval.dense.chunking.method=recursive_token \
  --set retrieval.dense.chunking.max_tokens=512 \
  --set retrieval.dense.chunking.overlap_tokens=64 \
  --set retrieval.dense.chunking.minimum_fill_ratio=0.5 \
  --output-root /kaggle/working/results
```

Audit Khoản boundaries before running the structure treatment:

```python
!python scripts/audit_clause_parser.py \
  --corpus data/legal-corpus.parquet \
  --seed 42 \
  --samples-per-status 30 \
  --output-dir /kaggle/working/results/khoan_parser_audit_seed42
```

Then run Khoản-aware dense retrieval with the same E5 budget, overlap, and max
aggregation as the fixed-window baseline:

```python
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_e5_khoan_512_64_seed42 \
  --set retrieval.dense.chunking.method=khoan \
  --set retrieval.dense.chunking.max_tokens=512 \
  --set retrieval.dense.chunking.overlap_tokens=64 \
  --output-root /kaggle/working/results
```

After the Khoản run, audit Điểm boundaries and run the Điểm-plus-Khoản-lead-in
treatment without changing the model or retrieval settings:

```python
!python scripts/audit_point_parser.py \
  --corpus data/legal-corpus.parquet \
  --seed 45 \
  --sample-law-count 50 \
  --exclude-summary /kaggle/working/results/khoan_parser_audit_50laws_seed44/summary.json \
  --samples-per-status 30 \
  --output-dir /kaggle/working/results/diem_parser_audit_seed45

!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_e5_diem_512_64_seed42 \
  --set retrieval.dense.chunking.method=diem \
  --set retrieval.dense.chunking.max_tokens=512 \
  --set retrieval.dense.chunking.overlap_tokens=64 \
  --output-root /kaggle/working/results
```

Use CLI flags for model trials instead of duplicating configuration files.
Evaluate BGE-M3 with the non-structural fixed token-window control before the
legal-structure treatments:

```bash
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_bge_m3_token_window_512_64_seed42 \
  --model-name BAAI/bge-m3 \
  --query-prefix "" \
  --document-prefix "" \
  --max-seq-length 512 \
  --encode-batch-size 4 \
  --devices 2 \
  --set retrieval.dense.chunking.method=token_window \
  --set retrieval.dense.chunking.max_tokens=512 \
  --set retrieval.dense.chunking.overlap_tokens=64 \
  --output-root /kaggle/working/results
```

Then keep the 512/64 budget fixed and switch only to the winning Điểm
representation:

```bash
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_bge_m3_diem_512_64_seed42 \
  --model-name BAAI/bge-m3 \
  --query-prefix "" \
  --document-prefix "" \
  --max-seq-length 512 \
  --encode-batch-size 4 \
  --devices 2 \
  --set retrieval.dense.chunking.method=diem \
  --set retrieval.dense.chunking.max_tokens=512 \
  --set retrieval.dense.chunking.overlap_tokens=64 \
  --output-root /kaggle/working/results
```

Only after that run, test BGE-M3's longer context with 1,024/128. This second
run changes both encoder and token-window budget relative to E5 and is therefore
a context-length ablation, not a model-only comparison:

```bash
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_bge_m3_diem_1024_128_seed42 \
  --model-name BAAI/bge-m3 \
  --query-prefix "" \
  --document-prefix "" \
  --max-seq-length 1024 \
  --encode-batch-size 4 \
  --devices 2 \
  --set retrieval.dense.chunking.method=diem \
  --set retrieval.dense.chunking.max_tokens=1024 \
  --set retrieval.dense.chunking.overlap_tokens=128 \
  --output-root /kaggle/working/results
```

If a Kaggle T4 reports CUDA out-of-memory at 1,024 tokens, lower only
`--encode-batch-size` from 4 to 2 or 1; do not change chunk size, overlap, split,
or retrieval depth in that retry.

## Hybrid retrieval

The initial hybrid run combines BM25 and Điểm-aware dense retrieval with
equal-weight WRRF (`rrf_k=60`). The portable config uses regex/NFKC BM25 and E5;
if its matching dense embeddings already exist, the validated cache is reused:

```python
!python scripts/run_experiment.py \
  --config configs/hybrid.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --train-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --experiment-name hybrid_regex_e5_diem_wrrf60_equal_seed42 \
  --output-root /kaggle/working/results
```

To use BGE-M3 while retaining Điểm 512/64 and the same fusion settings:

```python
!python scripts/run_experiment.py \
  --config configs/hybrid.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --train-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --experiment-name hybrid_regex_bge_m3_diem_wrrf60_equal_seed42 \
  --model-name BAAI/bge-m3 \
  --query-prefix "" \
  --document-prefix "" \
  --max-seq-length 512 \
  --encode-batch-size 4 \
  --devices 2 \
  --output-root /kaggle/working/results
```

The best measured VnCoreNLP BM25 treatment can replace the portable regex
branch by additionally setting `word_segmenter=vncorenlp` and an explicit local
JAR path. Keep the same path used by the standalone VnCoreNLP run so the hybrid
comparison changes only fusion:

```python
  --set retrieval.bm25.preprocessing.tokenizer.word_segmenter=vncorenlp \
  --set retrieval.bm25.preprocessing.tokenizer.vncorenlp_jar_path=/kaggle/working/VnCoreNLP-1.2.jar
```

Each hybrid artifact includes fused `predictions.jsonl` and the untouched BM25
and dense lists in `component_rankings.jsonl`. Tune weights or `rrf_k` only
after preserving the equal-weight run as the fixed starting point.

Pseudo-relevance feedback is disabled by default. Against the selected main
pipeline, the unchanged config is the no-feedback control. Enable RM3 with:

```python
  --set retrieval.bm25.feedback.enabled=true
```

Enable dense Rocchio with:

```python
  --set retrieval.dense.feedback.enabled=true
```

Provide both overrides for the combined treatment and use a distinct
`--experiment-name` for all four states. Keep VnCoreNLP, WRRF20, weights,
candidate depths, epoch-3 checkpoint, and output policy fixed.
`retrieval_feedback.json` records pseudo-relevant article IDs and RM3 expansion
terms. Do not reuse a LightGBM model or threshold across feedback treatments;
feedback changes sparse/dense ranks and the downstream feature distribution.

`notebooks/archive/kaggle_rm3_rocchio_ablation.ipynb` is retained as a deferred
Rocchio-only screen, not an active recommended run. It defines a controlled
2-by-3 grid: 3 or 5 positive feedback aids and beta 0.10, 0.25, or 0.50, with
alpha 1, gamma 0, no negative feedback, and first-pass depth 20. If profiling
later makes the treatment viable, it can train a fresh five-fold query-grouped
OOF LightGBM per feature distribution and select public/private submissions.
Do not launch this full notebook under the current Kaggle budget: the owner
reported on 2026-10-09 that the Rocchio screen stalled before a usable result.

RM3 was removed from this Kaggle screen after the owner-reported 2026-10-08 log
showed the conservative profile reproduce the selected epoch-3 threshold/
fallback-2 F2 `0.608053`, while the following larger profile did not begin
loading the reranker until `43212` seconds, approximately the 12-hour Kaggle
limit. The log did not contain an OOM exception; this is recorded as an observed
runtime failure, not proof of memory exhaustion. Larger RM3 profiles require a
separate efficiency treatment, such as feedback-term document-frequency caps or
optimized postings scoring, before being restored to the end-to-end notebook.
Rocchio likewise requires a retrieval-only timing profile before its six-case
screen is restored. Until then both feedback mechanisms remain disabled and the
selected baseline is epoch-3 reranking plus OOF LightGBM and query-adaptive
score-gap final-set selection. The global-threshold and fixed-top-2 outputs
remain required controls.

### Multi-dense peer notebook

`notebooks/kaggle_multi_dense_weight_selection.ipynb` accepts a list of peer
dense components without a privileged anchor. The active two-model run is
`mainguyen9/vietlegal-harrier-0.6b` plus
`intfloat/multilingual-e5-large`. Each entry carries its own model identifier,
prompt/prefix, remote-code setting, cache, and WRRF weight. The corresponding reusable configuration is
`configs/hybrid_reranker_multi_dense.yaml`.

The notebook retrieves standard BM25 and every configured dense component once.
It evaluates five normalized WRRF weight combinations offline from those fixed
top-100 component lists, selects by fixed top-3 development F2 with Recall@50 as
the tie-break, and runs the epoch-3 direct-chunk reranker only for that winner.
It then evaluates a newly trained query-grouped five-fold OOF LightGBM and
creates public/private fixed-top-2 and threshold/fallback-2 submissions both
with and without LightGBM, plus the selected LightGBM score-gap submissions.
Output is archived at
`/kaggle/working/multi_dense_results.tar.gz`.

Preflight requires two GPUs with at least 14 GiB each, at least 24 GiB system
RAM, and 12 GiB free working storage. Artifact directories include a fingerprint
of the active model/prefix list, sparse variant, and selected weights so a later
component list or BM25 variant cannot silently reuse OOF or test outputs from an
earlier case.

The first run keeps `BM25_VARIANT='bm25'`. After that control completes, set
only `BM25_VARIANT='bm25_plus'` in the same multi-dense notebook; the default
delta is `1.0`, and both variant and delta enter the run/cache identity. This
order tests the stronger sparse scorer without confounding it with the change
from one dense model to two. BM25+ is not omitted as a candidate improvement;
it is deliberately sequenced as the next isolated ablation.

Offline article-rank weight selection is an efficient screening approximation:
the final direct-chunk fusion is reconstructed by the selected weights only in
the reranker run. Therefore the weight-grid scores are not evidence that the
same ordering holds after chunk fusion and reranking; the final dev metrics are
reported separately.

For multi-dense reranking, candidate generation deliberately finishes before
the epoch-3 cross-encoder loads. The runner then releases dense encoder models,
the two corpus embedding matrices, cached query embeddings, and full chunk
lists while retaining the selected top-50 chunk candidates and saved component
rankings. The log line
`[experiment] Releasing retrieval resources before reranker load` (or the
corresponding `[inference]` line) is required before reranker checkpoint
materialization. If an older revision remains at partial RoBERTa weight loading
for hours, stop it and rerun with this lifecycle repair; it is not productive
reranking work.

For the active packed-stream batch-256 run, both development evaluation and
public/private inference use a validated JSON retrieval handoff followed by
Python process-image replacement before the cross-encoder is constructed.
This makes process exit, rather than allocator cleanup alone, the CUDA ownership
boundary. The inference handoff preserves each named split's first-stage ranks
and scores, retrieved chunks, component rankings, feedback, and retriever
metadata; public and private query coverage is validated after restart.

The active multi-dense notebook uses three independent retrieval caches:
VnCoreNLP BM25, VietLegal-Harrier corpus embeddings, and multilingual-E5-large
corpus embeddings. A sparse cache stores the ordered `aid` array, retained term
vocabulary, postings offsets/positions/frequencies, retained-vocabulary
document lengths, average document length, vocabulary statistics, corpus
fingerprint, and preprocessing/index key. It does not store query rankings.
A dense cache stores the ordered chunk-parent `aid` mapping in validated
metadata and the float32 corpus embedding matrix, together with corpus and
encoder/chunker fingerprints. It does not store model weights, reconstructed
chunk objects/text, query embeddings, exact-search results, fused rankings, or
reranker scores. Consequently, a dense cache hit avoids corpus encoding but
still requires deterministic rechunking for validation/alignment, model loading
for query encoding, matrix loading, and exact search. First-stage component
rankings are separate experiment artifacts, not part of the embedding cache.

The preferred attachment layout is one Kaggle Dataset containing the three
already-extracted cache roots and their sibling manifests: BM25S,
VietLegal-Harrier, and multilingual-E5-large. The active notebook can read
these immutable files directly under `/kaggle/input`; this avoids tar
extraction, working-disk duplication, and accidentally attaching multiple
versions of the same slug. Do not rebuild dense embeddings merely because the
reranker batch size changed: reranker batching is absent from the dense cache
identity.

Dense `embeddings.npy` files are loaded read-only with NumPy memory mapping.
This preserves the exact float32 matrix while avoiding an eager anonymous-RAM
copy before FAISS prepares its search index. The active notebook no longer runs
a separate full-corpus `--require-existing` preparation pass for a populated
attached cache; the actual experiment remains fail-closed with `require_hit`
and performs the authoritative corpus/chunker/encoder/alignment validation.
New embedding caches also record and verify the matrix SHA-256 checksum. Older
caches without this optional field remain readable for backward compatibility;
their corpus, encoder/chunker identity, shape, dtype, and ordered `aid` mapping
are still validated as before.

The remaining cache limitation is deterministic rechunking. Each embedding
cache stores the matrix and ordered parent `aid` mapping, but not the 205,898
serialized `Chunk` records needed for chunk-level retrieval and reranking. A
cache hit must therefore rerun the shared BGE-M3 Khoản–Điểm 448/64 tokenizer
pipeline. The next cache-format revision should add one shared, immutable chunk
table keyed by the ordered raw-corpus fingerprint and chunker key, then make
both dense embedding caches reference its fingerprint. Store chunk ID, parent
`aid`, `law_id`, chunk index/type, text, character offsets, and token count in a
portable Parquet or JSONL artifact. Keep the current rechunk-and-validate path
as a backward-compatible fallback. Only after that implementation and parity
tests should Harrier and E5 caches be rebuilt; rebuilding them in the current
format provides no scientific or runtime benefit.

Once BM25S target `0` and shared-chunk target `8` are attached alongside the
existing seven dense caches, retrieval caching is operationally complete for
the current protocol. A healthy multi-dense run must show a BM25S cache hit,
one or more shared chunk-cache hits, and embedding-cache hits for Harrier and
multilingual-E5. Do not add query, FAISS-index, reranker-score, or LightGBM
prediction caches without a measured bottleneck. The next useful execution
boundary is a fingerprinted selected-candidate artifact followed by a separate
reranking process, because that enables failure recovery and controlled
reranker ablations without repeating retrieval.

## Hybrid plus cross-encoder reranking

After fixing the hybrid candidates, run the zero-shot BGE reranker over the top
50 candidates with Điểm plus Khoản lead-in representations:

```python
!python scripts/run_experiment.py \
  --config configs/hybrid_reranker.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --train-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --experiment-name hybrid_vncorenlp_bge_m3_diem_wrrf20_d08_s02_reranker50_seed42 \
  --devices 2 \
  --set retrieval.bm25.preprocessing.tokenizer.word_segmenter=vncorenlp \
  --set retrieval.bm25.preprocessing.tokenizer.vncorenlp_jar_path=/kaggle/working/VnCoreNLP/VnCoreNLP-1.2.jar \
  --output-root /kaggle/working/results
```

Replace the VnCoreNLP JAR path with the verified path used by the standalone
BM25 run. The BGE-M3 dense corpus cache should hit when model, Điểm 512/64
chunking, prefixes, and corpus are unchanged. Cross-encoder scores are not
cached in this first implementation and must be computed for every
query–representation pair. If T4 memory is insufficient, lower only
`reranking.batch_size` while retaining the same candidates and token budgets:

```python
  --set reranking.batch_size=4
```

Most optimized reranker configs use continuous pair batching:

```yaml
reranking:
  batching: global_stream
  outer_pair_batch_size: 1024
  batch_size: 128
```

The active multi-dense end-to-end recovery control is intentionally
`batching: per_query` with `batch_size: 8`; this matches the conservative
pre-optimization execution mode while validating the retrieval-to-reranking
process handoff. `outer_pair_batch_size` bounds each group passed to the scorer, while
`batch_size` is the actual Hugging Face/GPU mini-batch. Keeping the outer size a
multiple of the neural batch size avoids introducing a partially filled batch
at every outer boundary. `global_stream` may combine the tail of one query with
the beginning of the next mini-batch, but restores every score to its original
query before aggregation. Use `--set reranking.batching=per_query` only as a
parity/debug control. The selected direct-chunk, 512-token two-T4 configuration
uses batch 128 after the measured benchmark reported 3.08 times batch-8
throughput, exact sampled top-1/top-3 parity, and 3,659 MiB peak allocated
memory. This setting does not transfer automatically to the 1,024-token
article-rechunk configs, which retain their separately safe batches.
`run_info.json` records batching mode, outer size, neural size, scored-pair
count, and scorer-call count.

For the smallest useful batch-size decision, run
`notebooks/kaggle_reranker_batch_benchmark.ipynb`. It loads the selected
checkpoint once, builds ten deterministic question-length-stratified query
groups with 50 real legal Khoản/Điểm candidates each, warms up batch sizes 8,
64, and 128, and rotates
their execution order for at least four repeats. It stops after four when all
throughput CVs are at most 2%; otherwise it continues until each treatment
accumulates 60 measured seconds, capped at seven repeats. The
artifact reports median pairs/second, coefficient of variation, peak allocated
CUDA memory, score drift, and a reranker-only extrapolation to the 21,900-pair
upper bound (`438 * 50`). The extrapolation excludes checkpoint loading,
retrieval, fusion, chunk preparation, and evaluation. Treat it as sufficient
for choosing the neural batch size when throughput CV is at most 10%, sampled
top-1 agreement against batch 8 is 100%, mean top-3 overlap is at least 99%, mean rank
correlation is at least 0.999, and the candidate has safe memory headroom; otherwise
extend the parity sample or retain batch 8. FP16 logits may vary slightly when
the batch shape changes, so maximum absolute logit drift is recorded as a
diagnostic rather than used alone as the acceptance criterion.

The result directory contains reranked `predictions.jsonl`, frozen
`first_stage_predictions.jsonl`, component rankings, resolved config, metrics,
split IDs, and both retriever and reranker metadata in `run_info.json`.
The CLI also prints fixed top-1 through top-5 metrics directly below the Kaggle
cell and saves them in `metrics_by_output_k.json`. Reranker runs additionally
print and save the best development-tuned score thresholds with top-1 and top-2
fallbacks in `metrics_by_score_threshold.json`; this post-processing does not
invoke the model again.

To test direct evidence reranking, keep the same models and switch only the
candidate source. Dense supplies 100 chunk candidates, chunk-level WRRF selects
50, the cross-encoder scores those 50 query–chunk pairs, and max aggregation to
articles happens afterward:

```python
!python scripts/run_experiment.py \
  --config configs/hybrid_reranker.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --train-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --experiment-name hybrid_bge_m3_diem_wrrf20_rerank_retrieved_chunks50_seed42 \
  --devices 2 \
  --set retrieval.bm25.preprocessing.tokenizer.word_segmenter=vncorenlp \
  --set retrieval.bm25.preprocessing.tokenizer.vncorenlp_jar_path=/kaggle/working/VnCoreNLP/VnCoreNLP-1.2.jar \
  --set reranking.candidate_source=retrieved_chunks \
  --set reranking.retrieved_chunk_depth=100 \
  --set reranking.candidate_pool_size=50 \
  --set reranking.max_length=1024 \
  --set reranking.batch_size=8 \
  --output-root /kaggle/working/results
```

This reuses the dense corpus cache but performs one additional exact query-to-
chunk search. Its Recall@K is coverage of the selected chunk pool after unique
`aid` projection, not the standard article-level hybrid Recall@K.

## Kaggle overnight verification suite

`notebooks/archive/kaggle_overnight_experiments.ipynb` is the archived thin Run-All launcher for
an approximately eight-hour T4x2 verification window. It uses explicit
`!python scripts/...` commands, validates the attached Parquet files, installs
the dense and Vietnamese extras, and runs these cases in order:

The setup intentionally retains the simple Kaggle workflow: clone, `%cd` into
the repository, install extras with `!python -m pip`, and invoke repository
entry points as `!python scripts/...`. The preflight reads the query Parquet
directly instead of importing `drill_ir` into the already-running notebook
kernel; all project code continues to execute in fresh script subprocesses.

1. VnCoreNLP BM25;
2. BGE-M3 fixed token-window 512/64 dense control;
3. BGE-M3 Điểm 512/64 dense treatment;
4. the selected WRRF20 hybrid first stage;
5. Điểm reranking with pair/document budgets 1024/768;
6. a reranker document-budget control that freezes the dense Điểm 512/64 first
   stage and pair max 1,024 while lowering only reranker oversize fallback from
   768 to 512;
7. an end-to-end efficiency treatment that retains natural Điểm units, lowers
   only their oversize token-window fallback ceiling from 512 to 448 for dense
   retrieval, and uses reranker pair/document budgets 512/448;
8. an optional reranker-structure control that retains the same dense Điểm
   448/64 first stage and changes only the reranker representation to fixed
   token windows, started only if the preceding cells finish within six hours;
9. direct reranking of 50 retrieved/fused Điểm chunks followed by max
   aggregation to `aid`, without reopening full candidate articles.

Each cell skips a case whose `metrics_by_output_k.json` already exists, displays
its fixed-k and threshold tables immediately, and preserves completed results if
a later case is interrupted. Dense cache reuse makes the hybrid and reranker
first stages inexpensive after the matching Điểm dense run. Cross-encoder pair
scores are not checkpointed, so an interrupted reranker case itself must still
restart. `DiemChunker.max_tokens=448` is a ceiling, not a fixed chunk size:
Điểm plus Khoản lead-in representations at or below 448 tokens remain intact,
and only oversized units use overlapping token-window fallback. Case 7 can
therefore change dense candidates only for units above that ceiling, while also
changing reranker context relative to case 5; interpret it as an end-to-end
efficiency treatment, not as a pure reranker-length ablation. The final cell
builds `overnight_scoreboard.csv` and archives the full
results directory as `/kaggle/working/drill_overnight_results.tar.gz`.

The independent hierarchical structure suite is
`notebooks/archive/kaggle_khoan_diem_experiments.ipynb`. It mirrors the same controls,
split, models, WRRF parameters, output-k sweep, reranker budgets, direct-evidence
case, and optional token-window reranker control, but selects `khoan_diem` for
the structure-aware dense and reranker stages. Its reranker base configuration
is `configs/hybrid_reranker_khoan_diem.yaml`, and its final archive is
`/kaggle/working/drill_khoan_diem_results.tar.gz`. Keep the original notebook
results as the `diem` comparison rather than overwriting them.

Less common leaves can be overridden with repeatable typed flags, for example
`--set retrieval.dense.search.query_batch_size=32`. Explicit convenience flags
take precedence over `--set`, and the final resolved configuration is written to
the result directory.

`configs/dense.yaml` uses `devices: auto`. With two visible Kaggle T4 GPUs,
Sentence Transformers receives both `cuda:0` and `cuda:1` for parallel corpus
and query encoding; with one GPU or CPU it falls back deterministically. Set
`devices: 2` only when failure on fewer devices is preferred over fallback.

Both path arguments are optional. The CLI resolves data in this order:

1. an explicitly supplied path;
2. `data/<filename>` for local execution;
3. the known `wedev-vlsp` Kaggle path;
4. a recursive lookup under `/kaggle/input`, accepted only if exactly one file
   with the expected name exists.

Therefore, after attaching the Kaggle dataset, this shorter command is valid:

```bash
!drill-validate
```

The command prints the resolved corpus and query paths before reading them. If
Kaggle mounts more than one matching dataset version, pass explicit paths to
avoid an ambiguous or scientifically incorrect selection.

The repository scripts remain supported without installation:

```bash
!python scripts/validate_data.py \
  --corpus /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --queries /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet
```

If `ModuleNotFoundError: drill_ir.data` appears after pulling an older revision,
verify that `src/drill_ir/data/` exists in the Kaggle checkout and pull the
revision containing the root-anchored `/data/` ignore rule. The previous
unanchored `data/` rule accidentally excluded the Python data package from Git.

## Hardware profiles

Experiments should expose hardware choices in configuration. Support at least:

```yaml
runtime:
  accelerator: auto       # cpu, cuda, or auto
  devices: auto           # 1, 2, or auto
  mixed_precision: false  # set explicitly per experiment
  batch_size: null        # stage-specific value should be recorded
  num_workers: 2
```

The preferred Kaggle profile should use both available NVIDIA T4 GPUs where the
stage benefits from parallelism. Likely candidates are corpus encoding,
cross-encoder training, and large reranking batches. BM25, data parsing, metric
calculation, and small inference jobs are normally CPU-bound or too small to
benefit and should not acquire GPUs unnecessarily.

Multi-GPU support must not silently change the scientific result. Preserve
input order and the mapping from vector positions to `aid`, gather distributed
outputs deterministically, avoid duplicate samples introduced by distributed
samplers, and record world size, per-device batch size, effective global batch
size, precision mode, GPU model, and runtime.

Use the simplest suitable strategy:

- independent data-parallel encoding/reranking shards for inference workloads;
- distributed data parallel for training when training is actually required;
- a deterministic single-GPU fallback for debugging and environments with one
  GPU;
- CPU fallback for correctness tests and stages that do not require CUDA.

Do not assume two GPUs are present merely because the config requests them.
Validate availability at startup and fail with a clear message or use an
explicitly configured fallback policy.

## Paths and portability

Do not hard-code Kaggle mount paths in core modules. Pass dataset, cache, model,
and output paths through CLI arguments or configuration. Environment-specific
path resolution may live in a thin notebook cell or launcher script.

The current Kaggle dataset locations are:

```text
/kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet
/kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet
```

Store these values in a Kaggle-specific data config or CLI arguments, not as
defaults embedded in loaders. A local config can point to the future local
`data/` directory while invoking the same experiment command.

Corpus embeddings and model caches may be reused between runs when their
metadata matches the dataset checksum, preprocessing version, model identifier,
and relevant encoding configuration. Never reuse an unverified cache.

The implemented dense cache keys corpus embeddings by a SHA-256 fingerprint of
ordered `(aid, law_id, text)` values plus the document-encoding configuration.
Query prefixes are deliberately excluded because queries are not stored in the
corpus cache. Each run records cache hit/miss and embedding shape in
`run_info.json`.

## Reranker fine-tuning on Kaggle

Before committing GPU hours to the full corpus, run
`notebooks/archive/kaggle_smoke_finetune_reranker.ipynb`. It uses one synthetic query,
one positive, five negatives, one GPU, and one optimizer step. Passing verifies
the FlagEmbedding/CUDA/tokenization/backpropagation/model-save path only; it
does not exercise retrieval, hard-negative mining, two-GPU DDP, or provide a
scientific metric.
The notebook also contains a level-2 check with two query groups, near-budget
passages, one two-GPU DDP optimizer step, checkpoint reload, and finite-logit
inference validation. Even level 2 remains a plumbing test rather than a
retrieval or quality experiment.

`notebooks/archive/kaggle_finetune_reranker_448.ipynb` is the archived Run-All launcher for the
first supervised trial. It validates the 512/448 pair budget, mines five hybrid
hard negatives per seed-42 training query, fine-tunes
`BAAI/bge-reranker-v2-m3` for two epochs with the official FlagEmbedding
encoder-only trainer, and evaluates the final checkpoint on the untouched
438-query development split. Core mining logic remains in
`src/drill_ir/reranking/training.py`; the notebook calls the thin
`scripts/prepare_reranker_training.py` entry point.

Mining writes `reranker_train_448.jsonl`, a separate audit JSONL, and metadata
that must report 1,752 written training queries and 438 excluded development
queries before training starts. The final archive includes the model,
train-data provenance, resolved evaluation config, predictions, fixed-cutoff
metrics, and newly tuned threshold metrics. Do not reuse a zero-shot score
threshold with the fine-tuned checkpoint.

`notebooks/archive/kaggle_finetune_khoan_diem_dual_candidate.ipynb` is the archived controlled
follow-up for the best fixed zero-shot setting: Khoản–Điểm 448/64 with a
512/448 reranker budget. It mines one train-only dataset and fine-tunes one
shared checkpoint, then evaluates that identical checkpoint with
`article_rechunk` and `retrieved_chunks` candidate flows. This isolates the
inference candidate flow from independently trained weights. Both branches
still max-aggregate reranker chunk scores to unique article IDs; they differ in
whether top-50 selection occurs before or after article aggregation.
The notebook pins FlagEmbedding 1.4.2, verifies the final model config and
weights before evaluation, retains epoch checkpoint directories, and includes
the complete model directory among its working artifacts. Do not archive the
entire work directory: it may include dense embedding caches, the final model,
duplicate epoch checkpoints, and result artifacts, and one observed Kaggle run
created a 3.6-GB partial gzip before exhausting disk space and preventing
papermill from saving the executed notebook. Package metrics/config/predictions
separately, then package only the selected checkpoint if it must be downloaded.
The archived notebook's final cell now creates only
`drill_finetuned_khoan_diem_dual_metrics.zip`; it does not tar `WORK`.
Its final cells select the
candidate flow and threshold/fallback only from the development results, infer
the official public and private JSON files with the selected fine-tuned
checkpoint, and produce both fixed top-2 and threshold submission ZIPs. A
separate validation cell checks the ZIP member name, payload keys, complete and
unique query-ID coverage, non-empty predictions, and integer article IDs before
the artifacts are archived.

`notebooks/archive/kaggle_finetune_fpt_article_bce.ipynb` is the false-negative-aware
recovery experiment. It calls `scripts/prepare_fpt_reranker_training.py` to
mine full-article labelled pairs using the published FPT.AI top-30, skip-first-5,
random-semi-hard policy, then calls `scripts/train_cross_encoder_bce.py` through
two-GPU `torchrun`. The training command uses an explicit PyTorch DDP loop with
the Hugging Face one-logit reranker and pointwise BCE rather than the earlier
FlagEmbedding grouped objective. The notebook first runs that exact path for
one optimizer step on both GPUs; the full run starts only after the smoke model
update and provenance are completed. Smoke weights are deliberately not saved
to avoid consuming several gigabytes before the real run. The notebook
checks the 1,752/438 split invariant, exposes deterministic audit samples before
training, retains both epoch checkpoints, and compares matched zero-shot,
epoch-1, and epoch-2 development results. It intentionally does not load the
public or private test JSON: test inference remains a separate action after a
fine-tuned checkpoint actually beats its matched zero-shot control.

This reproduction is expected to be substantially slower than the earlier
grouped run. A typical query contributes one or more positive pairs plus five
or ten negative pairs, while the effective batch remains 16 and the sequence
budget increases to 1,024. Do not estimate its completion time from the old
110-steps-per-epoch run. Preserve the mining JSONL/metadata and both epoch
checkpoints so an interrupted evaluation does not require retraining.

The BCE trainer saves epoch 1 and epoch 2 normally, then exposes epoch 2 at the
final model root through hard links when supported. This keeps the convenient
final-model path without a third physical copy of the multi-gigabyte weights.
Before the full run, it also refuses to start unless estimated space for both
epoch snapshots plus a 2 GiB reserve is available.
The FPT notebook's final ZIP contains only metrics, predictions, scoreboard,
provenance, and mining metadata/audit; it excludes checkpoints, model caches,
dense embeddings, and the large optimization JSONL.

`notebooks/archive/kaggle_finetune_main_retrieved_chunks_bce.ipynb` is aligned with the
selected main pipeline. It calls `scripts/prepare_direct_chunk_bce_training.py`,
trains pointwise BCE at pair length 512, and compares matched zero-shot,
epoch-1, and epoch-2 checkpoints on the same direct top-50 retrieved chunks.
Negative chunks exactly match the inference candidate unit; positive chunks
are weakly selected from gold articles and must be inspected through the audit
JSONL. Its final archive contains only results, metadata, and audit records—not
embedding caches or duplicate model checkpoints. Use this notebook for the
main fine-tuning decision and retain the article FPT notebook as a control.

`notebooks/archive/kaggle_finetune_main_retrieved_chunks_bce_epochs3_4.ipynb` is the
controlled longer-training follow-up. It preserves the same seed, mined-pair
protocol, pointwise-BCE objective, effective batch size, learning rate, and
inference pipeline, but trains for four epochs. The current trainer retains one
loadable model snapshot for every completed epoch; this notebook evaluates
epochs 3 and 4 alongside the same zero-shot control. Its work directory, result
case names, scoreboard, and small metrics archive are separate from the
two-epoch run. Do not compare intermediate checkpoints as a single learning
curve across the two notebooks because their linear schedules have different
total step counts.

The BCE trainer supports exact epoch-boundary continuation with
`--resume-from-checkpoint`. Every epoch checkpoint is permanently usable for
inference, while only the latest keeps the much larger optimizer, scheduler,
scaler, and per-rank RNG state. Resume validates the dataset SHA-256, world
size, batch/accumulation settings, optimizer hyperparameters, sequence length,
seed, steps per epoch, and scheduler horizon before loading state.

To stop after epoch 2 but preserve the exact schedule for a later epoch-4
continuation, predeclare the horizon on the first run:

```python
!torchrun --nproc_per_node 2 scripts/train_cross_encoder_bce.py \
  --model-name-or-path BAAI/bge-reranker-v2-m3 \
  --train-data /kaggle/working/direct_chunks_bce_train.jsonl \
  --output-dir /kaggle/working/reranker_training \
  --max-length 512 --epochs 2 --scheduler-total-epochs 4 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --seed 42
```

Then resume from its latest checkpoint, keeping the same output directory when
writable so all epoch models remain together:

```python
!torchrun --nproc_per_node 2 scripts/train_cross_encoder_bce.py \
  --model-name-or-path BAAI/bge-reranker-v2-m3 \
  --resume-from-checkpoint /kaggle/working/reranker_training/checkpoint-STEP \
  --train-data /kaggle/working/direct_chunks_bce_train.jsonl \
  --output-dir /kaggle/working/reranker_training \
  --max-length 512 --epochs 4 --scheduler-total-epochs 4 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --seed 42
```

Without a predeclared longer scheduler horizon, an epoch-2 run cannot be called
an exact resume to epoch 4; loading its weights is only a warm start.

`notebooks/kaggle_epoch3_lightgbm_submission.ipynb` consumes the selected
epoch-3 checkpoint from Kaggle Models and the archived epoch-3 development run.
It reproduces the fixed strong pipeline for public/private queries, persists
the reranker, first-stage, sparse, dense, and fused-chunk inputs, evaluates
pointwise LightGBM with five query-grouped out-of-fold splits, fits the final
classifier on all labeled development candidates, and packages both fixed
top-2 and OOF-threshold/fallback-2 submissions. The notebook expects the model
at `/kaggle/input/models/lmnhnht125/drill-reranker-bce-epoch3/pytorch/default/1/checkpoint-epoch3`
and the prior metrics archive under
`/kaggle/input/datasets/lmnhnht125/drill-epoch3-4-experiment-artifacts`.

The notebook defaults `ADAPTIVE_SELECTION` to the selected `"score_gap"`
final-set policy and additionally packages
`public_lightgbm_score_gap_results.zip` and
`private_lightgbm_score_gap_results.zip`; use `"disabled"` to reproduce only
the original fixed-top-2 and global-threshold outputs. Both modes retain those
original outputs. The score-gap parameters come from pooled OOF development
tuning, while `oof_metrics.json` separately records its nested-OOF selector
estimate and a clearly labeled, non-deployable oracle-prefix diagnostic.
The notebook validates the exact ZIP list implied by this mode, so score-gap
runs expect six submission ZIPs and disabled runs expect four.

If an older aligned run failed with
`DistributedDataParallel object has no attribute preprocess`, pull the repaired
training script and rerun the training cell in the same Kaggle session. Do not
rerun mining when the validated `direct_chunks_bce_train.jsonl`, audit JSONL,
and metadata still exist: the exception happened at optimizer step zero. An
empty or partial model directory is not treated as completion because the
notebook requires final config, weights, provenance, and two valid epoch
checkpoint directories.

## Standalone dense-model screening

The completed dense screen can be reproduced with
`notebooks/archive/kaggle_dense_standalone_screening.ipynb` after building or
uploading a verified dense cache archive. Set only `DENSE_TARGET` from 1 through
7 according to the notebook's model table. The notebook discovers either an
expanded cache dataset or `retrieval_cache_<slug>.tar`, extracts archives under
Kaggle working storage, and runs `scripts/run_experiment.py` with
`configs/dense_screening.yaml`. `require_hit=true` prevents an accidental full
corpus re-encoding when the archive, corpus, model prefix, or chunk protocol is
wrong.

Each run is dense-only: no BM25, weighted RRF, or reranker. It writes the usual
config, split IDs, ranking, metrics, and provenance, then packages them as one
downloadable `dense_screening_<slug>.tar.gz`. Use the same target number in
seven cloned notebooks to run the models concurrently. Model weights are still
loaded to encode development queries; the cache removes corpus encoding, which
is the dominant cost. Weight tuning should happen later from saved component
rankings so changing weighted-RRF coefficients does not rerun any encoder.

## Public/private submission inference

The submission runner fits the configured corpus index once, infers the public
and private JSON query sets separately, then validates and packages each set.
For the current Kaggle mounts, run the selected fixed configuration as follows:

```python
!python scripts/run_submission_inference.py \
  --config configs/hybrid_reranker.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --public-test /kaggle/input/datasets/lmnhnht125/wedev-vlsp-test/public_test.json \
  --private-test /kaggle/input/datasets/lmnhnht125/wedev-vlsp-test/private_test.json \
  --devices 2 \
  --output-dir /kaggle/working/submission_results
```

This creates `public_results.zip` and `private_results.zip`, each containing
only root-level `results.json`. It also retains the full ranked JSONL files,
first-stage rankings for reranker runs, resolved `config.yaml`, and
`run_info.json` under the output directory. The config's `retrieval.output_k`
controls the submitted list length; override it explicitly with `--output-k`
only when applying a policy already selected on train/dev.

To additionally package the dev-selected retrieved-chunks threshold policy,
select the matching candidate source and pass the recorded threshold explicitly:

```python
!python scripts/run_submission_inference.py \
  --config configs/hybrid_reranker.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --public-test /kaggle/input/datasets/lmnhnht125/wedev-vlsp-test/public_test.json \
  --private-test /kaggle/input/datasets/lmnhnht125/wedev-vlsp-test/private_test.json \
  --devices 2 \
  --output-k 2 \
  --score-threshold 4.015625 \
  --threshold-fallback-k 2 \
  --set reranking.candidate_source=retrieved_chunks \
  --output-dir /kaggle/working/submission_results
```

The same inference pass then produces both fixed top-2 ZIPs and additional
`public_threshold_results.zip` / `private_threshold_results.zip`. A threshold
must only be used with the exact model, representation, score precision, and
candidate-source configuration on which it was selected.

For post-competition sensitivity analysis, `--score-thresholds` accepts several
values and creates one ZIP per threshold while reusing the same retrieval and
reranker pass. It is mutually exclusive with the backward-compatible singular
`--score-threshold`. Output names encode both threshold and fallback, for
example `private_threshold_4p234375_fallback2_results.zip`. Sweeping test
submissions is leaderboard probing, not development tuning; report the frozen
development threshold `4.234375` separately from neighboring sensitivity runs.

For `retrieved_chunks`, the runner processes candidate generation atomically per
test split: `retrieve(public)` immediately precedes `retrieve_chunks(public)`,
then the same sequence is applied to private. This ordering is required because
WRRF chunk fusion consumes sparse component ranks from the matching retrieval
batch. All public/private chunk candidates are completed before the
cross-encoder loads, after which both sets are reranked and every threshold ZIP
is derived from the persisted rankings. A regression test with disjoint public
and private qids protects this ordering.

## Query-level retrieval diagnostics

After a labeled development run, generate diagnostics without rerunning any
retriever or reranker:

```python
!python scripts/analyze_query_retrieval.py \
  --result-dir /kaggle/working/multi_dense_selection/selected_dev/RUN_NAME \
  --queries /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --query-ids /kaggle/working/multi_dense_selection/selected_dev/RUN_NAME/split_dev_qids.json \
  --ranking lightgbm_oof=/kaggle/working/multi_dense_selection/lightgbm_oof/RUN_ID/oof_predictions.jsonl \
  --output-dir /kaggle/working/multi_dense_selection/query_analysis
```

`stage_metrics.csv` compares all discovered components, fusion, actual reranker
input, epoch-3, and optional later rankings. `hard_queries.csv` orders queries
by exact contribution to the macro Recall@50 deficit.
`query_diagnostics.jsonl` retains gold ranks and attribution, and `summary.json`
contains compact aggregate results. The multi-dense notebook runs this audit
automatically and packages the outputs and required development rankings.
Use `notebooks/kaggle_query_recall50_analysis.ipynb` to inspect those artifacts
separately. It accepts one compatible `multi_dense_results.tar.gz`, performs no
model inference, displays only focused tables, and produces the lightweight
`query_recall50_diagnostics.tar.gz` download.

The multi-dense notebook also writes `wall_times.json` using only notebook-level
`time.perf_counter()` calls around subprocesses. It reports coarse wall time for
cache preparation when needed, first-stage retrieval, selected dev retrieval
plus reranking, LightGBM, query analysis, and public/private inference. It does
not import or inspect CUDA, RAM, models, or retrieval internals and therefore
cannot affect model execution. The file is included in
`multi_dense_results.tar.gz`.

To run the fixed-budget distinct-article candidate ablation without changing
the tracked baseline, override the explicit policy leaf:

```python
!python scripts/run_experiment.py \
  --config configs/hybrid_reranker_multi_dense.yaml \
  --corpus-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet \
  --train-path /kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet \
  --output-root /kaggle/working/best_per_aid_ablation \
  --devices 2 \
  --set retrieval.hybrid.chunk_candidate_selection=best_per_aid
```

Keep `reranking.candidate_pool_size=50` and
`reranking.retrieved_chunk_depth=100` fixed for the first comparison. The
resolved config and retriever metadata record the selected policy. This test
maximizes distinct-article coverage within the existing chunk-scored pool; it
is not the later treatment that guarantees the article-level fused top 50.

The tracked `notebooks/kaggle_multi_dense_weight_selection.ipynb` is currently
configured as this full treatment: `best_per_aid`, packed-stream reranking,
batch size 256, and a separate
`/kaggle/working/multi_dense_best_per_aid` output root. It writes
`multi_dense_best_per_aid_results.tar.gz` and compares its development metrics
against the measured 2026-10-28 `top_chunks` packed-stream batch-256 baseline.
Switching `CHUNK_CANDIDATE_SELECTION` back to `top_chunks` restores the control.
