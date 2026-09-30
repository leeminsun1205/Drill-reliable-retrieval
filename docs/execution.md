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

`notebooks/kaggle_overnight_experiments.ipynb` is the thin Run-All launcher for
an approximately eight-hour T4x2 verification window. It uses explicit
`!python scripts/...` commands, validates the attached Parquet files, installs
the dense and Vietnamese extras, and runs these cases in order:

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
