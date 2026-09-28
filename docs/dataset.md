# Dataset

## Raw legal corpus

The current Kaggle input path is:

```text
/kaggle/input/datasets/lmnhnht125/wedev-vlsp/legal-corpus.parquet
```

Each corpus row represents a legal document:

```text
id: int
law_id: str
content:
    list[
        {
            aid: int,
            content_Article: str
        }
    ]
```

The normal retrieval unit is an article, not the complete document. Flatten the
corpus deterministically to:

```text
aid
law_id
content_Article
```

Document-level indexing is allowed only for an explicitly documented
document-level experiment.

## Query dataset

The current Kaggle training path is:

```text
/kaggle/input/datasets/lmnhnht125/wedev-vlsp/train.parquet
```

```text
qid: int
question: str
relevant_laws: list[int]
```

Despite its name, `relevant_laws` contains ground-truth article IDs (`aid`). The
inspected training Parquet file uses `qid`, matching the official paper and the
submission schema.

The task website example supplied to this project uses `id`, while both the
official shared-task paper and the inspected file use `qid`. Treat the actual
Parquet schema as authoritative. The loader accepts either spelling for
portability, normalizes it to `qid` internally, and rejects rows containing both
with conflicting values. A normal retrieval result maps the query identifier to
a ranked list of `aid` values.

An article is relevant when the legal question can be answered Yes/No and that
answer is entailed by the meaning of the article. Relevance is therefore based
on answer-bearing legal entailment, not merely topical similarity.

## Preprocessing requirements

- Never modify original raw Parquet files in place.
- Make preprocessing deterministic.
- Preserve `aid` without lossy conversion or renumbering.
- Verify uniqueness assumptions explicitly rather than silently relying on them.
- Persist split IDs and reuse them across methods.
- Verify that every input query ID is preserved exactly once in predictions and
  submissions.

For every derived dataset, document its source/version, transformation and
filtering, random seed if applicable, output schema, and creation command or
configuration.

Raw data, derived corpora, checkpoints, and embeddings are local artifacts and
must not be committed unless explicitly requested and appropriately reviewed.

## Verified local snapshot

The files inspected on 2026-09-28 contain:

- 2,157 legal documents and 59,636 articles;
- unique, contiguous document IDs `0..2156` and article IDs `0..59635`;
- 2,190 labeled queries with unique, non-contiguous `qid` values;
- 2,939 label entries covering 2,505 unique articles;
- no ground-truth article IDs missing from the corpus.

Known anomalies are one empty document, eight blank article texts, and three
queries whose `relevant_laws` contain duplicate IDs. The loader deduplicates gold
IDs while preserving first occurrence. Blank articles are reported by validation
and excluded from text indexes; none of the eight blank articles is a training
gold label.

Article lengths are highly skewed: median 835 characters, 99th percentile 8,835,
and maximum 255,092. Dense encoders and cross-encoders must therefore record and
test truncation or chunking behavior rather than relying on implicit tokenizer
defaults.

## Official split and evidence-set statistics

The organizer reports 2,190 labeled training queries, 312 public-test queries,
and 627 private-test queries. The average number of relevant articles is 1.34,
1.31, and 1.32 respectively; the corresponding maxima are 9, 4, and 5. These
published values confirm that DRiLL is a variable-size evidence-set retrieval
task rather than a top-1 classification task.

The organizer also groups queries into five legal domains: Economics and
Finance (EF), State Management and Law (SL), Society, Culture and Education
(SCE), Infrastructure and Development (ID), and Science and Technology (ST).
Reported totals are 1,048, 1,052, 379, 431, and 219 queries respectively. The
current Parquet files do not include a domain column or an official query-to-
domain mapping. Domain-wise evaluation must therefore remain disabled until a
reproducible mapping is obtained; inferred domain labels must not be presented
as organizer annotations.

Source: Vuong et al., DRiLL Shared Task 2025 overview and its extended journal
version, linked from `docs/competition_review.md`.
