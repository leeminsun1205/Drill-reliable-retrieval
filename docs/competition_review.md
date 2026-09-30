# VLSP 2025 DRiLL competition review

This document records publicly verifiable competition results, representative
methods, and released code. It distinguishes organizer facts from participant
self-reports and from this repository's local experiments.

## Official evaluation split

The organizer reports one shared legal corpus of 59,636 articles and three query
subsets:

| Subset | Queries | Labels available to participants | Intended use |
|---|---:|---|---|
| Train | 2,190 | Yes | Training and local development |
| Public test | 312 | Hidden | Public leaderboard during development |
| Private test | 627 | Hidden | Final ranking |

Sources: the organizer's [VLSP 2025 paper](https://aclanthology.org/2025.vlsp-1.16/)
and [extended journal article](https://doi.org/10.25073/2588-1086/vnucsce.6463).

The competition did not publish an official labeled development subset inside
the 2,190 training queries. Teams therefore created their own validation split
or tuned indirectly on the public leaderboard. The official private set must not
be treated as a development set.

This repository currently makes a deterministic random 80/20 split of the
official training set: 1,752 train and 438 development queries with seed 42. The
development IDs are persisted per run. This split is a local research protocol,
not an organizer split, and is not directly comparable with public/private
leaderboard scores. It is currently random rather than domain- or label-count-
stratified; future changes must preserve the existing split as a named baseline
for comparability.

The organizer reports a variable number of relevant articles per query. Train,
public test, and private test average 1.34, 1.31, and 1.32 relevant articles,
with maxima of 9, 4, and 5. This makes evidence completeness and final-set
selection central to the task: a correct top-1 article may still be incomplete.

The five reported domains are Economics and Finance (EF), State Management and
Law (SL), Society, Culture and Education (SCE), Infrastructure and Development
(ID), and Science and Technology (ST), with 1,048, 1,052, 379, 431, and 219
queries across all splits. EF and SL therefore dominate the benchmark. The
extended overview reports SCE as the weakest average top-10-team domain and
identifies a domain-adaptation gap. The released local Parquet snapshot has no
domain field, so this repository cannot yet reproduce domain-wise evaluation.

## Private-test results

The final private leaderboard reported by the organizer is:

| Rank | Team | Precision | Recall | F2 |
|---:|---|---:|---:|---:|
| 1 | EDM | 0.6773 | 0.7394 | 0.7261 |
| 2 | FPT IS | 0.6222 | 0.7181 | 0.6966 |
| 3 | CUDO | 0.5097 | 0.7653 | 0.6955 |
| 4 | Berry | 0.5509 | 0.7097 | 0.6710 |
| 5 | IUH.TD | 0.4153 | 0.7605 | 0.6521 |
| 6 | brain_rot | 0.4714 | 0.7172 | 0.6495 |
| 7 | AImba | 0.4086 | 0.7498 | 0.6425 |
| 8 | BK_DRiLL | 0.4329 | 0.7077 | 0.6280 |
| 9 | Engineers | 0.3147 | 0.7478 | 0.5864 |
| 10 | villageai | 0.3799 | 0.6332 | 0.5587 |
| 11 | NaiveNotNice | 0.2769 | 0.7472 | 0.5578 |
| 12 | Organizer two-stage baseline | 0.3740 | 0.6253 | 0.5512 |
| 19 | Organizer BM25 baseline | 0.2265 | 0.3830 | 0.3365 |

Source: [extended organizer paper, Table 7](https://doi.org/10.25073/2588-1086/vnucsce.6463).
These are published competition values, not measurements from this repository.

The public leaderboard had a different ordering: FPT IS, CUDO, and brain_rot
were the top three, while eventual winner EDM ranked seventh. This change is a
warning against selecting a method primarily from public-leaderboard feedback.

### Final output-size policies behind reported private scores

The organizer leaderboard reports Precision, Recall, and F2 but does not report
a common final `top_k` for participant systems. Participant papers show that
candidate depth and final output size must not be conflated: top-50 retrieval or
top-10 reranking can still feed a thresholded, variable-size answer set.

| Private result | Team/system | Published final selection policy |
|---|---|---|
| F2 0.7261, rank 1 | EDM | Variable-size output: a validation-selected F2 threshold over pointwise LightGBM probabilities; explicitly not fixed top-k. The threshold value and output-size distribution are not published. |
| F2 0.6966, rank 2 | FPT IS | Variable-size pipeline: retain BGE-reranked chunks above 0.7, deduplicate to article IDs, cluster scores, then apply Qwen2.5-72B entailment filtering. The exact final count/fallback and output-size distribution are not published. |
| F2 0.6955, rank 3 | CUDO submitted checkpoint | Variable-size output: inspect the reranked top 10, keep scores above 0.99, and fall back to top 2 only when no candidate passes. Thus `top-2` is a fallback, not the normal fixed output size. |
| F2 0.6521, rank 5 | ViDRILL | Variable-size output: threshold 0.75 with a fallback policy. Public-test ablations select hybrid retrieval depth 30 and top-40 fallback; the paper does not unambiguously establish that this exact fallback produced the private submission. |
| F2 0.5864, rank 9 | Engineers | Variable-size output: reranker threshold 0.85 plus a hybrid fallback; the published summary does not provide a fixed final count. |
| F2 0.5512/0.5515 | Organizer two-stage baseline | Fixed top 2. The extended organizer ablation reports F2 0.5496, 0.5515, and 0.4615 for fixed top 1, 2, and 5 respectively. |

The papers available for other leaderboard teams do not establish a sufficiently
precise final cardinality policy, so no fixed top-k should be imputed to them.
For this repository, fixed top-1/top-2/top-3 cutoff analysis is a necessary
baseline, but a dev-calibrated threshold with a deterministic non-empty fallback
is the closer comparison to the strongest participant systems. Thresholds are
model-specific and must be retuned after reranker fine-tuning; values such as
0.99, 0.85, and 0.75 are not transferable across uncalibrated score scales.

## Representative approaches

### EDM — rank 1

EDM used a multi-stage learning-to-rank system. Candidate retrieval combined
BM25, dense retrieval, HyDE, automatically generated titles, and explicit
inter-article reference signals. It progressively reduced roughly 500 candidates
to 100 and then 10. LambdaMART integrated retrieval, reranker, similarity, and
statistical features; LLM pairwise and collective chain-of-thought scores were
then fused with previous signals by pointwise LightGBM and a validation-selected
threshold. The system is especially relevant to this project's structure and
cross-reference research question.

EDM explicitly states that it did not fine-tune the large pretrained dense or
reranker models; only the lightweight learning-to-rank modules were retrained.
BGE-M3 embedding similarity and BGE-v2-M3 reranker scores were frozen input
features for LambdaMART/LightGBM rather than task-fine-tuned neural retrieval
components. Thus the winning result is evidence for feature-rich candidate
generation and learned score fusion, not evidence that neural retriever
fine-tuning was necessary.

Source: [EDM working-notes paper](https://aclanthology.org/2025.vlsp-1.19/).
No public GitHub repository was found in the paper or targeted searches as of
2026-09-28.

### FPT IS — rank 2

FPT IS followed a data-centric pipeline: Qwen2.5-7B-Instruct generated missing
titles and summaries; legal texts were hierarchically chunked, with 2,048-token
chunks and 1,024-token overlap for very long leaves; BGE-M3 embeddings were
stored in Milvus; the top 50 chunks were reranked with BGE-reranker-v2-m3 and
filtered at 0.7; deduplication, score clustering, and Qwen2.5-72B entailment
judgment formed the final stages.

Source: [data augmentation and hierarchical chunking paper](https://aclanthology.org/2025.vlsp-1.18/).
No confirmed public GitHub repository was found.

### CUDO — rank 3

CUDO enriched corpus articles with document hierarchy/titles, then used hybrid
BM25 plus BGE-M3 retrieval and BGE-reranker-v2-m3. Their documented setup used
BM25 parameters `k1=0.5`, `b=0.5`, a hybrid weight of 0.6, top-100 candidates,
top-10 reranked candidates, and a score threshold above 0.99 with top-2 fallback.
The reranker was fine-tuned for two epochs using hard negatives mined after
skipping the first five high-scoring non-gold candidates.

Source: [simple two-stage framework paper](https://aclanthology.org/2025.vlsp-1.20/).
No confirmed public GitHub repository was found.

### IUH.TD / ViDRILL — rank 5

ViDRILL combined BM25 with multilingual dense retrievers including E5-Instruct
and GTE, followed by BGE reranking. The work explored dual-level chunking and
hard-negative training. Its authors state that the official competition pipeline
did not use the repository's later LLM query-rewriting/reasoning extensions.

Sources: [ViDRILL paper](https://aclanthology.org/2025.vlsp-1.17/) and
[public ViDRILL repository](https://github.com/xndien2004/ViDrill).

### BK_DRiLL — rank 8

The organizer summary describes an initial GTE-multilingual-E2 retriever,
ViRanker reranking to top 10, then multi-round voting/debate using Qwen2.5-14B
and Qwen2.5-32B. A public repository by the rank-eight participant contains the
released implementation and checkpoint links.

Source: [rank-eight public repository](https://github.com/NgJaBach/VLSP-2025-Challenge-on-DRiLL-The-challenge-of-Deep-Retrieval-in-the-expansive-Legal-Landscape).

### Engineers — rank 9

Engineers used VnCoreNLP plus Pyserini BM25, fine-tuned multilingual-e5-large,
and normalized linear hybrid fusion (`0.3 * sparse + 0.7 * dense`). Both sparse
and dense retrieval returned top 100, followed by a fine-tuned
BGE-reranker-v2-m3 trained with five hard negatives. Final selection used a 0.85
reranker threshold and hybrid fallback.

Source: [Engineers public repository](https://github.com/duongkstn/flow-vlsp-drill-legal-retrieval).

### Public top-11 pipeline

One public repository self-reports a top-11 private finish but does not clearly
state the official team label. It uses chunking, BM25, BGE-M3 dense retrieval,
rank/score fusion, BGE-reranker-v2-m3 over the top 100, and optional
Qwen2.5-32B KEEP/DROP filtering.

Source: [Vietnamese Legal Document Retrieval repository](https://github.com/chisngooo/Vietnamese-Legal-Document-Retrieval).

## Confirmed public code

The following repositories contain code and explicitly identify themselves as
VLSP 2025 DRiLL systems:

1. [xndien2004/ViDrill](https://github.com/xndien2004/ViDrill) — ViDRILL/IUH.TD.
2. [NgJaBach/VLSP-2025-Challenge-on-DRiLL...](https://github.com/NgJaBach/VLSP-2025-Challenge-on-DRiLL-The-challenge-of-Deep-Retrieval-in-the-expansive-Legal-Landscape) — rank-eight participant.
3. [duongkstn/flow-vlsp-drill-legal-retrieval](https://github.com/duongkstn/flow-vlsp-drill-legal-retrieval) — Engineers.
4. [chisngooo/Vietnamese-Legal-Document-Retrieval](https://github.com/chisngooo/Vietnamese-Legal-Document-Retrieval) — self-reported top-11 system; official team label not confirmed from the README.

No public organizer baseline repository was found. Public availability does not
by itself guarantee that every leaderboard-producing checkpoint, generated
artifact, or exact environment is included; each repository must be audited
before using it as a reproducibility reference.

## Main lessons for this project

1. Candidate recall matters: successful systems generally retrieve 50–500
   candidates before expensive ranking.
2. Final selection matters as much as ranking: fixed top-k often lost to tuned
   thresholds with a fallback policy.
3. Structure helps, but evidence must be isolated: titles, hierarchy, and
   cross-references appeared in strong systems, supporting controlled ablations
   rather than assuming gains.
4. Hard negatives consistently improved rerankers.
5. Very long articles require an explicit chunking/truncation policy.
6. Public/private rank shifts make private-test tuning scientifically unsafe.
7. Several top solutions require large LLMs or external reconstruction steps;
   they are useful upper references but are not appropriate first baselines for
   a reproducible T4x2 research pipeline.
8. Hybrid retrieval and reranking are competition-standard strong baselines,
   not the proposed scientific contribution. The contribution must be isolated
   in structure/reference-aware ablations over an unchanged strong baseline.

## Competition-comparability boundary

The official competition restricted external data, required eligible pretrained
models to have been publicly available before the competition cutoff, prohibited
closed-source models, and required reproducibility. Experiments may deliberately
relax these rules for post-competition research, but their artifacts must record
that choice and must not be described as directly competition-comparable.

This distinction will later be represented explicitly in experiment metadata;
for now it is a documented requirement rather than an implemented config field.

## Method decomposition and reusable design patterns

Participant systems can be understood as five separable decisions:

1. **Representation:** raw article, title-enriched article, hierarchy-preserving
   chunk, summary, or reference-aware representation.
2. **Candidate generation:** BM25, dense retrieval, hybrid fusion, HyDE, or
   explicit reference expansion. This stage should optimize candidate recall.
3. **Candidate discrimination:** cross-encoder, learning-to-rank, or LLM judge.
   This stage should improve precision without losing relevant candidates.
4. **Final-set selection:** fixed top-k, calibrated threshold, fallback rule, or
   learned pointwise decision. DRiLL evaluates a set, so this is part of the
   method rather than output formatting.
5. **Supervision:** gold positives, random/in-batch negatives, BM25/dense hard
   negatives, and near-positive filtering.

The organizer BM25 baseline exercises only candidate generation and fixed-k
selection. The organizer two-stage baseline adds hard-negative cross-encoder
training. Engineers and CUDO add hybrid candidates and threshold selection.
FPT IS changes representation through hierarchy-aware chunking and adds an LLM
judge. EDM modifies all five decisions and learns the final selection boundary.

### Why candidate recall and final F2 differ

A deep candidate list can have excellent Recall@100 or Recall@500 but extremely
low precision if submitted directly. Candidate depth is therefore an internal
budget, not the number of final predictions. Rerankers cannot recover a gold
article absent from the candidate pool, while a high-recall pool alone performs
poorly on final macro-F2. Record both candidate recall and final set metrics.

### Why hard negatives matter

Random negatives are usually obviously unrelated and teach little. Hard
negatives share legal vocabulary or topic with the query but are not annotated
as relevant, forcing the model to learn answer-bearing entailment rather than
topic similarity. However, the highest-scoring unannotated articles may be
unlabeled positives. CUDO's decision to skip the first five non-gold hits before
sampling negatives is one practical defense against false-negative supervision.

### Why thresholding often beats fixed top-k

DRiLL queries have variable gold-set sizes. Fixed top-k always returns the same
number, creating false positives for single-article questions and insufficient
coverage for multi-article questions. A calibrated score threshold adapts output
size, but score scales can drift across models and domains. A robust design
therefore combines a dev-selected threshold with a deterministic fallback such
as top-1 or top-2 and reports the output-size distribution.

### Why title and hierarchy enrichment help

Many supplied `content_Article` values omit the provision title and surrounding
chapter context. Similar clause text can therefore be hard to distinguish.
Prepending deterministically extracted titles or hierarchy paths gives both
lexical and dense retrievers discriminative context. Generated titles may add
hallucinations, so deterministic extraction and generated enrichment must be
evaluated as separate treatments.

## Published chunking strategies

The DRiLL literature contains three relevant but distinct approaches:

| System | Dense retrieval unit | Long-text policy | Reranking unit | Article aggregation |
|---|---|---|---|---|
| EDM (rank 1) | Both article- and clause-level representations are reported; exact clause construction is unpublished | Not published | Article candidates with article/clause-derived features and BGE reranker scores | Learned fusion through LambdaMART/LightGBM; no isolated chunking ablation |
| FPT IS (rank 2) | Hierarchical leaves such as clauses or bullet points, enriched with their root-to-leaf path | A leaf over 4,000 tokens becomes 2,048-token windows with 1,024 overlap | Retrieved chunks | Deduplicate by `aid`, retaining the maximum chunk score before later filtering |
| CUDO (rank 3) | Organizer article boundary enriched with reconstructed document titles/hierarchy | Not reported as intra-article splitting | Query–article pairs | Already article-level |
| ViDRILL (rank 5) | Legal clauses capped at 450 **characters**, without overlap | Oversized clauses are split into smaller coherent segments | Separate chunks up to 2,000 **characters**, with line-break overlap | Source `aid` is retained, but the complete inference-time aggregation rule is unpublished |
| Public top-11 implementation | Recursive windows of 400 whitespace-delimited words | 50-word overlap; separator order is blank line, line, space, then character fallback | Retrieved chunks | Preserve `aid`; full final policy depends on later pipeline stages |

The units above are not directly interchangeable: the papers variously report
characters, model tokens, or whitespace-delimited words. In particular,
ViDRILL's 450-character retrieval chunks are much shorter than a 512-token
window, whereas FPT IS allows structural leaves up to 4,000 tokens and only
then applies overlapping windows. Their results therefore do not identify a
single universally best chunk length.

1. **FPT IS hierarchical chunking.** A generated article title is the root;
   articles are split into smaller units such as clauses or bullet points; tree
   leaves become chunks and receive their root-to-leaf hierarchy path as
   metadata. A leaf over 4,000 tokens is further divided into 2,048-token
   windows with 1,024-token overlap. Retrieval takes top 50 chunks, and
   post-processing deduplicates by article ID while retaining the highest score.
   This is the closest published design to Khoản/Điểm-aware retrieval, but the
   paper does not publish its Vietnamese boundary parser, nested-amendment
   handling, exact chunk schema, or a public implementation.

   This should not be confused with the winning EDM system. EDM's paper also
   reports clause segmentation and a hierarchical corpus containing laws,
   articles, and clauses. Its LambdaMART features include `article_idx`,
   `clause_idx`, `is_law`, `is_clause`, query/document token lengths, and
   length ratios. However, EDM does not publish the clause-boundary algorithm,
   a point-level split, maximum chunk length, overlap, oversized-clause policy,
   or chunk-to-article score aggregation. The evidence therefore supports
   **article plus clause representations**, but not the stronger claim that EDM
   used the proposed Article -> Khoan -> Diem -> tokenizer-window procedure.
   EDM also combines this representation with generated descriptions, HyDE,
   reference expansion, neural reranking, LambdaMART/LightGBM, and large-LLM
   relevance scoring, so its final score cannot be attributed to chunking
   alone. Its paper provides no article-only versus clause-level ablation.

2. **ViDRILL dual-level chunking.** Reranking uses chunks up to 2,000 characters
   with overlap based on line-break boundaries. Dense retrieval uses independent
   legal clauses capped at 450 characters without overlap; oversized clauses are
   split again into smaller coherent segments. Multiple chunks retain their
   source `aid`. For positive training data, BGE-M3, E5-Instruct, and GTE scores
   rank chunks belonging to a gold aid, and the highest-scoring chunk is selected
   as the representative positive. The paper does not specify a Khoản/Điểm
   parser, clause-lead propagation, nested quotation handling, or the complete
   final chunk-to-aid aggregation rule.

3. **Public top-11 fixed-window implementation.** The released `utils/chunk.py`
   uses LangChain `RecursiveCharacterTextSplitter`, `len(text.split())`, a
   400-word budget, 50-word overlap, and separators `\n\n`, `\n`, space, then
   character fallback. It preserves `aid` and assigns `{aid}_{chunk_index}`.
   This is reproducible and useful as a non-structural baseline, but it does not
   explicitly model Khoản, Điểm, hierarchy paths, or amendment nesting.

CUDO's “document-based chunking with titles” reconstructs external document
hierarchy and maps titles back to the organizer's article-level chunks; it is
primarily article enrichment rather than intra-article Khoản/Điểm splitting.

These approaches support three controlled baselines for this repository:
fixed-window chunks, clause-level chunks, and point-level chunks with clause
context. Our proposed deterministic parser differs by explicitly retaining raw
offsets and parse status, separating outer amendment structure from quoted
inner provisions where possible, and using tokenizer windows only as fallback.

### Why cross-reference features are promising

A query may require a provision that defines a concept or exception referenced
by the initially matched article. Reference-aware expansion can improve recall,
while reference-edge features can help a ranker select supporting provisions.
Unbounded expansion introduces many false positives. The controlled experiment
should keep the same hybrid candidates and reranker, changing only deterministic
reference extraction/expansion with an explicit edge budget.
