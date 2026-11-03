# DRiLL Reliable Retrieval

Research code for reproducible Vietnamese legal information retrieval,
initially centered on the VLSP 2025 DRiLL benchmark.

The current research goal is to evaluate legal structure and explicit article
references against a strong, reproducible retrieval and reranking baseline.

## Current pipeline and measured result

The current practical submission baseline uses VnCoreNLP BM25,
VietLegal-Harrier, and multilingual-E5-large fused by WRRF (`k=10`, weights
`0.15/0.65/0.20`), fixed token windows 448/64, reference expansion off, a
pool-50 cross-encoder, LightGBM `enhanced_v2` (61 features), and
multiclass–Ridge final-set selection. The reranker is fixed-window pointwise-MIL
epoch 1, checkpoint `1234` (three adaptation epochs trained with a five-epoch
scheduler horizon).

Owner-reported Codabench F2 is `0.6978` public (`1027650`) and `0.6825` private
(`1027647`). This is the current practical baseline because the project
prioritizes private performance. The prior checkpoint-1208 run remains the
public-score record at `0.7055`; these results and the post-hoc selection
caveat are documented in [`docs/experiments.md`](docs/experiments.md).

## Repository layout

```text
configs/             Experiment configurations
data/                Local datasets (ignored by Git)
docs/                Research protocol and data/evaluation documentation
notebooks/           Exploration, visualization, and error analysis
demo/                Optional Streamlit interface for the Kaggle demo
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
  --set retrieval.bm25.preprocessing.tokenizer.word_segmenter=pyvi
```

VnCoreNLP additionally requires
`retrieval.bm25.preprocessing.tokenizer.vncorenlp_jar_path` to point to its local
JAR. The default remains the dependency-light regex tokenizer used by the
recorded baseline. Underthesea and character/whitespace tokenizer modes were
removed after the initial ablation to keep the maintained matrix focused.

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
- [Nine report tables and slide mapping](tables_experiment.md)
- [Full 21-table experiment catalog](tables_experiment_catalog.md)
- [Final report experiment plan](docs/final_report_plan.md)

Raw data, model checkpoints, embeddings, and large generated outputs must remain
outside Git.

Dataset access and competition registration are provided through the
[VLSP 2025 DRiLL shared-task website](https://vlsp.org.vn/vlsp2025/eval/drill).
