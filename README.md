# DRiLL Reliable Retrieval

Research code for reproducible Vietnamese legal information retrieval,
initially centered on the VLSP 2025 DRiLL benchmark.

The current goal is to establish strong lexical, dense, hybrid, and reranking
baselines before evaluating structure-aware and cross-reference-aware retrieval.

## Repository layout

```text
configs/             Experiment configurations
data/                Local datasets (ignored by Git)
docs/                Research protocol and data/evaluation documentation
notebooks/           Exploration, visualization, and error analysis
scripts/             Thin command-line entry points
src/drill_ir/        Reusable research implementation
tests/               Scientific correctness and regression tests
results/             Local experiment artifacts (created on demand; ignored)
```

`AGENTS.md` contains concise repository-wide operating constraints for coding
agents. Durable project knowledge is organized by topic under `docs/`; it is not
duplicated into an agent-specific memory file. The root README remains the
human-facing entry point.

## Installation and validation

From the repository root:

```bash
python -m pip install -e .
drill-validate --corpus data/legal-corpus.parquet --queries data/train.parquet
drill-run --config configs/bm25.yaml
```

For the optional dense-retrieval stack:

```bash
python -m pip install -e '.[dense]'
drill-run --config configs/dense.yaml
```

Optional Vietnamese word segmenters for BM25 are installed with:

```bash
python -m pip install -e '.[vietnamese]'
drill-run --config configs/bm25.yaml \
  --set retrieval.bm25.preprocessing.tokenizer.word_segmenter=underthesea
```

PyVi is selected with `pyvi`. VnCoreNLP also requires
`retrieval.bm25.preprocessing.tokenizer.vncorenlp_jar_path` to point to its
local JAR. The default remains the dependency-light regex tokenizer used by the
recorded baseline.

Stopword and document-frequency ablations are config-driven and disabled by
default. After selecting a word segmenter, use
`--set retrieval.bm25.preprocessing.tokenizer.stopword_preset=vi_legal_conservative`
or `--set retrieval.bm25.min_document_frequency=2` in a separate run.

The scripts in `scripts/` provide equivalent commands for environments where an
editable install is not desired.

`drill-validate` automatically checks the local `data/` paths and the configured
Kaggle input location. Explicit `--corpus` and `--queries` paths remain available
when multiple dataset versions are attached.

## Research documentation

- [Research plan](docs/research_plan.md)
- [Dataset](docs/dataset.md)
- [Evaluation](docs/evaluation.md)
- [Experiments and artifacts](docs/experiments.md)
- [Local and Kaggle execution](docs/execution.md)
- [Competition methods, results, and public code](docs/competition_review.md)

Raw data, model checkpoints, embeddings, and large generated outputs must remain
outside Git.

Dataset access and competition registration are provided through the
[VLSP 2025 DRiLL shared-task website](https://vlsp.org.vn/vlsp2025/eval/drill).
