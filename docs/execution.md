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

Use CLI flags for model trials instead of duplicating configuration files. For
example, BGE-M3 on T4x2 can reuse the dense base config while clearing the E5
prefixes:

```bash
!python scripts/run_experiment.py \
  --config configs/dense.yaml \
  --experiment-name dense_bge_m3_seed42 \
  --model-name BAAI/bge-m3 \
  --query-prefix "" \
  --document-prefix "" \
  --max-seq-length 1024 \
  --encode-batch-size 4 \
  --devices 2 \
  --output-root /kaggle/working/results
```

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
