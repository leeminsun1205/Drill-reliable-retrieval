# DRiLL Reliable Retrieval

Research code for reproducible Vietnamese legal information retrieval,
initially centered on the VLSP 2025 DRiLL benchmark.

The current goal is to establish strong lexical, dense, hybrid, and reranking
baselines before evaluating structure-aware and cross-reference-aware retrieval.

## Repository layout

```text
configs/             Experiment configurations
docs/                Research protocol and data/evaluation documentation
notebooks/           Exploration, visualization, and error analysis
scripts/             Thin command-line entry points
src/drill_ir/        Reusable research implementation
tests/               Scientific correctness and regression tests
results/             Small reproducibility artifacts; large outputs are ignored
```

The repository is currently being scaffolded. Commands for data preparation and
experiment execution will be documented as their implementations are added.

## Research documentation

- [Research plan](docs/research_plan.md)
- [Dataset](docs/dataset.md)
- [Evaluation](docs/evaluation.md)
- [Experiments and artifacts](docs/experiments.md)
- [Local and Kaggle execution](docs/execution.md)

Raw data, model checkpoints, embeddings, and large generated outputs must remain
outside Git.

Dataset access and competition registration are provided through the
[VLSP 2025 DRiLL shared-task website](https://vlsp.org.vn/vlsp2025/eval/drill).
