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
`notebooks/kaggle_khoan_diem_experiments.ipynb`. It mirrors the same controls,
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
`notebooks/kaggle_smoke_finetune_reranker.ipynb`. It uses one synthetic query,
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

`notebooks/kaggle_finetune_fpt_article_bce.ipynb` is the false-negative-aware
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

`notebooks/kaggle_finetune_main_retrieved_chunks_bce.ipynb` is aligned with the
selected main pipeline. It calls `scripts/prepare_direct_chunk_bce_training.py`,
trains pointwise BCE at pair length 512, and compares matched zero-shot,
epoch-1, and epoch-2 checkpoints on the same direct top-50 retrieved chunks.
Negative chunks exactly match the inference candidate unit; positive chunks
are weakly selected from gold articles and must be inspected through the audit
JSONL. Its final archive contains only results, metadata, and audit records—not
embedding caches or duplicate model checkpoints. Use this notebook for the
main fine-tuning decision and retain the article FPT notebook as a control.

`notebooks/kaggle_finetune_main_retrieved_chunks_bce_epochs3_4.ipynb` is the
controlled longer-training follow-up. It preserves the same seed, mined-pair
protocol, pointwise-BCE objective, effective batch size, learning rate, and
inference pipeline, but trains for four epochs. Because the trainer retains the
last two physical snapshots, this notebook labels and evaluates them as epoch 3
and epoch 4, alongside the same zero-shot control. Its work directory, result
case names, scoreboard, and small metrics archive are separate from the
two-epoch run. Do not compare intermediate checkpoints as a single learning
curve across the two notebooks because their linear schedules have different
total step counts.

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

If an older aligned run failed with
`DistributedDataParallel object has no attribute preprocess`, pull the repaired
training script and rerun the training cell in the same Kaggle session. Do not
rerun mining when the validated `direct_chunks_bce_train.jsonl`, audit JSONL,
and metadata still exist: the exception happened at optimizer step zero. An
empty or partial model directory is not treated as completion because the
notebook requires final config, weights, provenance, and two valid epoch
checkpoint directories.

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
