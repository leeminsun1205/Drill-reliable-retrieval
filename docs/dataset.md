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

### Internal article structure

The raw schema is structured only to `document -> article`. Each
`content_Article` is a flat string containing the complete article body; when
present, numbered clauses (Khoản) and lettered points (Điểm) remain embedded in
that string rather than appearing as nested fields. The article's own heading or
number is generally not included as a separate field, while occurrences such as
`Điều 11` inside the text usually denote cross-references.

A reproducible spot check on 2026-09-28 sampled 20 article IDs with Python
`random.Random(42)`: 16 contained heuristic numbered-clause markers and 9
contained heuristic lettered-point markers. Four short articles contained
neither. These regex counts are structural observations, not authoritative
Khoản/Điểm annotations; formatting varies and markers are frequently inline.

Consequently, fixed-token chunking may split a Khoản or Điểm. A future
structure-aware chunker must extract boundaries deterministically, retain
`chunk_id -> aid`, and keep extraction separate from article-level retrieval and
evaluation.

### Temporary chunk-granularity audit

On 2026-09-28, a temporary full-corpus audit compared article, heuristic Khoản,
and heuristic Điểm granularity. Điểm representations were formed as `Khoản
lead-in + Điểm text`, and every row retained `document_id`, `law_id`, `aid`, and
`article_position`. Length was approximated by `len(encoding_text.split())`;
these are whitespace-delimited word counts, not BGE-M3 tokenizer counts.

| Granularity | Chunks | Mean words | Chunks over 1,024 words | Aids with an over-limit chunk | Maximum words |
|---|---:|---:|---:|---:|---:|
| Article | 59,636 | 303.24 | 2,035 | 2,035 | 55,097 |
| Khoản | 207,012 | 86.53 | 479 | 384 | 12,403 |
| Điểm + Khoản lead-in | 349,904 | 61.74 | 167 | 108 | 9,440 |

All three temporary outputs covered all 59,636 aids, had unique chunk IDs, and
passed exact `original_text[char_start:char_end] == chunk_text` checks. The
heuristic parser found no Khoản boundary for 8,909 articles and therefore kept
those as article-level fallbacks. At point granularity, 148,339 chunks were
Khoản fallbacks because their Khoản had no detected Điểm.

The audit demonstrates that legal-structure splitting greatly reduces but does
not eliminate over-limit text. Any production chunker still needs a tokenizer-
window fallback for oversized Điểm, Khoản, unstructured articles, tables, or
appendices. The temporary files are under `/tmp/drill_chunk_analysis/` and are
not reproducibility artifacts; these measurements must be rerun with the actual
model tokenizer before reporting model-specific truncation rates.

### Fifty-document formatting audit

A deterministic `random.Random(42)` sample of 50 document rows contained 1,622
articles. A conservative heuristic recognized Khoản only at the article start or
after a blank-line separator, and Điểm only in list-like contexts within the
text. Results were:

- 1,427/1,622 articles (88.0%) had at least one conservative Khoản marker;
- 1,304/1,622 (80.4%) began with `1.` or `1)`;
- 758/1,622 (46.7%) had at least one contextual Điểm marker;
- all 50 documents contained some articles with Khoản markers, and 47/50
  contained some articles with Điểm markers;
- among articles with detected Khoản, 1,386/1,427 (97.1%) had a complete
  detected `1..n` sequence under the conservative parser;
- 28 articles longer than 200 whitespace-delimited words had no conservative
  Khoản boundary.

The 41 apparent non-sequential cases were frequently formatting misses rather
than genuinely missing legal clauses: examples contained glued text such as
`...2023.2.` or `2.Phát hành`, so a whitespace-dependent regex detected `1` and
`3` but missed `2`. Other exceptions included appendix/manual-style decimal
sections such as `4.1`, `4.2`, administrative tails, tables, and articles with an
unnumbered lead-in. Decimal-like patterns are also ambiguous with dates,
percentages, legal references, and document numbers.

The resulting parser rule should therefore be hierarchical and conservative:

1. take article boundaries from the supplied schema, never regex;
2. detect candidate Khoản markers using position and punctuation, then validate
   local numeric sequence rather than treating every number as a Khoản;
3. detect Điểm only inside the resolved Khoản span, using list context and the
   Vietnamese label sequence `a, b, c, d, đ, e, ...`;
4. preserve unnumbered lead-ins as context;
5. recognize decimal appendix sections separately from legal Khoản;
6. retain ambiguous text and fall back to tokenizer windows instead of forcing
   a structural annotation.

These figures measure formatting regularity under heuristics, not legal-boundary
precision or recall. A manually reviewed boundary sample is still required
before treating the parser as a research component.

A second `random.Random(43)` sample selected 50 documents after excluding all
document IDs from the first sample. It contained 1,498 articles and had zero
document overlap. The same conservative heuristic found:

- 1,362/1,498 articles (90.9%) with Khoản markers;
- 1,259/1,498 (84.0%) beginning with `1.` or `1)`;
- 692/1,498 (46.2%) with contextual Điểm markers;
- 1,299/1,362 detected Khoản sequences (95.4%) equal to global `1..n`;
- one article longer than 200 whitespace-delimited words without a conservative
  Khoản boundary;
- 49/50 documents with some Khoản markers and 49/50 with some Điểm markers.

Across the two disjoint samples (100 documents, 3,120 articles), 2,789 articles
(89.4%) had conservative Khoản markers, 2,563 (82.1%) began with `1.`/`1)`, and
1,450 (46.5%) had contextual Điểm markers. Of articles with detected Khoản,
2,685/2,789 (96.3%) matched a single global `1..n` sequence. In total, 99/100
documents had some detected Khoản structure and 96/100 had some detected Điểm
structure.

The second sample exposed a further exception: amending instruments embed quoted
provisions with their own Điều/Khoản/Điểm numbering. A flattened article can
therefore contain valid nested sequences such as `1, 2, 2, 3, ...`, or a quoted
replacement article followed by the next outer amendment item. Duplicate or
restarted numbering in these cases is not a formatting error. The parser must
not require one global sequence across the complete string; it should preserve
quotation/amendment spans where deterministically detectable, or mark the case
ambiguous and use a non-structural fallback. Other confirmed exceptions were
glued markers, unnumbered lead-ins, tables, administrative tails, and provisions
whose numbering begins above one.

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
- 2,157 distinct non-null, non-blank `law_id` values, with no duplicates in this
  snapshot;
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
