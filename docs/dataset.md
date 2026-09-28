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
id: int
question: str
relevant_laws: list[int]
```

Despite its name, `relevant_laws` contains ground-truth article IDs (`aid`). The
raw training-query identifier is named `id`; the official submission schema
renames this identifier to `qid`.

The task website example supplied to this project uses `id`, while the official
shared-task paper illustrates the training field as `qid`. Treat the actual
Parquet schema as authoritative. The loader must accept either spelling,
normalize it to `qid` internally, reject rows containing both with conflicting
values, and test the conversion explicitly. A normal retrieval result maps the
query identifier to a ranked list of `aid` values.

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
