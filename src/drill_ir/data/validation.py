"""Deterministic validation of DRiLL data invariants."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class ValidationReport:
    documents: int
    articles: int
    queries: int
    blank_articles: int
    empty_documents: int
    duplicate_aids: int
    duplicate_qids: int
    duplicate_gold_rows: int
    missing_gold_aids: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def validate_dataset(corpus_path: str | Path, query_path: str | Path) -> ValidationReport:
    corpus = pq.read_table(corpus_path).to_pylist()
    queries = pq.read_table(query_path).to_pylist()

    aids: list[int] = []
    blank_articles = 0
    empty_documents = 0
    for document in corpus:
        content = document["content"] or []
        empty_documents += not content
        for article in content:
            aids.append(int(article["aid"]))
            blank_articles += not (article["content_Article"] or "").strip()

    qid_key = "qid" if "qid" in queries[0] else "id"
    qids = [int(row[qid_key]) for row in queries]
    gold_rows = [list(map(int, row["relevant_laws"] or [])) for row in queries]
    gold_aids = {aid for row in gold_rows for aid in row}
    aid_set = set(aids)

    return ValidationReport(
        documents=len(corpus),
        articles=len(aids),
        queries=len(queries),
        blank_articles=blank_articles,
        empty_documents=empty_documents,
        duplicate_aids=len(aids) - len(aid_set),
        duplicate_qids=len(qids) - len(set(qids)),
        duplicate_gold_rows=sum(len(row) != len(set(row)) for row in gold_rows),
        missing_gold_aids=len(gold_aids - aid_set),
    )
