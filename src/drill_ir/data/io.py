"""Readers for the DRiLL corpus and query datasets."""

from collections.abc import Iterator
import json
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
    """Load labeled or unlabeled queries from Parquet or JSON.

    Official test JSON may use either ``qid`` or the legacy ``id`` field and
    may omit ``relevant_laws``. Missing labels are represented by an empty
    tuple; callers performing evaluation must continue to use labeled data.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError("Query JSON must contain a top-level array")
    else:
        table = pq.read_table(path)
        names = set(table.column_names)
        if not names & {"qid", "id"}:
            raise ValueError("Query data must contain either 'qid' or 'id'")
        if "question" not in names:
            raise ValueError("Query data must contain 'question'")
        rows = table.to_pylist()

    queries: list[Query] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every query row must be a mapping")
        if "qid" not in row and "id" not in row:
            raise ValueError("Query data must contain either 'qid' or 'id'")
        if "question" not in row:
            raise ValueError("Query data must contain 'question'")
        if "qid" in row and "id" in row and int(row["qid"]) != int(row["id"]):
            raise ValueError(f"Conflicting qid/id values: {row['qid']} != {row['id']}")
        raw_qid = row.get("qid", row.get("id"))
        qid = int(raw_qid)
        if qid in seen:
            raise ValueError(f"Duplicate query ID: {qid}")
        seen.add(qid)
        # Gold labels are set-valued; preserve first occurrence for stable output.
        relevant = tuple(
            dict.fromkeys(int(aid) for aid in (row.get("relevant_laws") or ()))
        )
        queries.append(
            Query(qid=qid, question=str(row["question"]), relevant_aids=relevant)
        )
    return queries
