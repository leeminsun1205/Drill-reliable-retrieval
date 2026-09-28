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

Here `qid` corresponds to the raw query field `id`, and `relevant_laws` contains
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
