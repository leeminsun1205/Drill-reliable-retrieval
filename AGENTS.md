# AGENTS.md

## Project scope

This is a reproducible Vietnamese legal information-retrieval research codebase,
initially centered on VLSP 2025 DRiLL. It is not primarily a chatbot.

The current question is whether legal structure and inter-provision references
improve retrieval over strong hybrid retrieval plus cross-encoder reranking.
Follow the baseline progression in `docs/research_plan.md`; do not introduce
complex RAG or LLM orchestration without an explicit scientific justification.

## Priorities

Use this order when trade-offs are necessary:

1. Correctness
2. Reproducibility
3. Experimental comparability
4. Simplicity
5. Runtime efficiency
6. Engineering elegance

Prefer the smallest design that answers the research question and permits a
controlled comparison.

## Sources of truth

Read the relevant document before changing methodology:

- `docs/research_plan.md`: questions, roadmap, and ablations.
- `docs/dataset.md`: schemas, IDs, and preprocessing.
- `docs/evaluation.md`: splits, metrics, and leakage prevention.
- `docs/experiments.md`: pipeline boundaries, configuration, and artifacts.

Do not silently contradict them. Report conflicts before changing the protocol.

## Critical data invariants

- Corpus and query datasets are separate.
- A raw corpus row is a document, but the normal retrieval unit is an article.
- Flatten the corpus to at least `aid`, `law_id`, and `content_Article`.
- Raw query field `id` identifies the query; official submissions call it `qid`.
- Query field `relevant_laws` contains ground-truth article IDs (`aid`).
- A prediction maps the query ID to a ranked list of `aid` values.
- Official `results.json` uses `qid` and `relevant_laws`; preserve every query
  exactly once.
- Preserve the mapping between vector/index positions and `aid`.
- Never modify raw data in place; create deterministic derived data.

Do not use document-level indexing unless an experiment explicitly studies it.

## Experimental integrity

- Fix random seeds and persist split IDs.
- Reuse the same split across compared methods.
- Tune on train/dev only; never use the Codabench private test as a dev set.
- Keep unrelated variables fixed in comparisons.
- Compare rerankers on the same first-stage candidates.
- Make structure and reference components independently disableable.
- Support ablations for every multi-component proposal.
- Derive reference edges only from explicit text, deterministic extraction, or
  documented annotation.
- Never fabricate data statistics, metrics, results, performance, citations, or
  legal annotations.
- Mark illustrative values as illustrative; do not present unexecuted results as
  measured results.

## Implementation boundaries

- Keep preparation, retrieval, fusion, reranking, evaluation, and analysis
  separable. Retrieval code must not evaluate itself.
- Keep reranking separate from first-stage retrieval.
- Put research parameters in explicit configuration.
- Select pipelines with a top-level `method` config field behind one experiment
  runner; new methods must reuse shared data, evaluation, and artifact stages.
- Record model identifiers; do not silently substitute models.
- Document normalization before combining heterogeneous raw scores.
- Put reusable logic in `src/drill_ir/`; keep `scripts/` as thin entry points.
- Notebooks may explore and visualize, but must not be the only implementation
  of core pipeline logic.
- Treat local, Kaggle, Colab, and GPU systems as execution environments for the
  same repository code.
- The primary workflow is local development -> Git push -> Kaggle execution.
  Keep commands and configs portable across both environments.
- Prefer configurations that can use both Kaggle T4 GPUs for suitable dense
  encoding, training, or reranking workloads, while retaining deterministic CPU
  and single-GPU fallbacks. Do not force multi-GPU onto CPU-bound stages.
- Prefer focused, mature dependencies. Do not add a full RAG framework for a
  small retrieval task.
- Reuse the documented layout. Explain why existing directories are insufficient
  before adding a new top-level directory.

## Data, artifacts, and Git safety

- Do not commit raw data, checkpoints, embeddings, secrets, API keys, or large
  generated artifacts.
- Prefer JSONL, JSON, YAML, or Parquet over pickle for permanent artifacts.
- Record useful provenance: config, seed, Git revision, model and dataset IDs,
  runtime, and hardware when relevant.
- Do not force-push, rewrite shared history, delete unrelated work, or modify
  unrelated files.
- Do not create commits unless explicitly requested.

Git is the source of truth; execution environments are not separate versions.

## Code and tests

Use Python unless another language is explicitly justified. Prefer small clear
functions, explicit arguments, useful type hints, descriptive names,
deterministic behavior, and minimal hidden state.

Prioritize scientific tests for corpus flattening, query parsing, ID mapping,
splitting, metrics, rank fusion, and reference expansion. Use hand-verifiable
metric examples. Run relevant tests and report failures.

## Working procedure

For every non-trivial task:

1. Inspect relevant files and implementation.
2. Read applicable source-of-truth documents.
3. Identify the smallest coherent change.
4. Implement without rewriting sound abstractions unnecessarily.
5. Run proportional tests or sanity checks.
6. Report changes, tests, and remaining limitations.

When adapting the older traffic-law chatbot, selectively reuse retrieval and
preprocessing. Do not migrate UI, serving, conversation management, or
application orchestration unless explicitly needed.

## Definition of done

A code change normally requires clear behavior/schema, relevant passing checks,
updated methodology documentation when applicable, and reported limitations.

A meaningful experiment additionally requires a recorded configuration, seed
where relevant, stable dataset/split ID, verified evaluation, predictions,
metrics, provenance, and enough information to reproduce the run. State any
intentional omission.
