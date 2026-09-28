"""Parquet readers for the DRiLL corpus and query datasets."""

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from .models import Article, Query


def iter_articles(path: str | Path, *, include_blank: bool = False) -> Iterator[Article]:
    """Yield flattened articles without mutating the document-level source."""
    parquet = pq.ParquetFile(path)
    required = {"law_id", "content"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Corpus is missing columns: {sorted(missing)}")

    for batch in parquet.iter_batches(columns=["law_id", "content"]):
        for row in batch.to_pylist():
            law_id = row["law_id"]
            for item in row["content"] or ():
                text = item["content_Article"] or ""
                if include_blank or text.strip():
                    yield Article(aid=int(item["aid"]), law_id=str(law_id), text=text)


def load_queries(path: str | Path) -> list[Query]:
    """Load labeled queries, accepting either qid or legacy id."""
    table = pq.read_table(path)
    names = set(table.column_names)
    id_columns = names & {"qid", "id"}
    if not id_columns:
        raise ValueError("Query data must contain either 'qid' or 'id'")
    if not {"question", "relevant_laws"} <= names:
        raise ValueError("Query data must contain 'question' and 'relevant_laws'")

    rows = table.to_pylist()
    queries: list[Query] = []
    for row in rows:
        if "qid" in row and "id" in row and row["qid"] != row["id"]:
            raise ValueError(f"Conflicting qid/id values: {row['qid']} != {row['id']}")
        raw_qid = row.get("qid", row.get("id"))
        # Gold labels are set-valued; preserve first occurrence for stable output.
        relevant = tuple(dict.fromkeys(int(aid) for aid in (row["relevant_laws"] or ())))
        queries.append(
            Query(qid=int(raw_qid), question=str(row["question"]), relevant_aids=relevant)
        )
    return queries
