"""Validation and packaging of official Codabench submissions."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package_submission(
    predictions_path: str | Path,
    expected_qids: set[int],
    destination: str | Path,
    *,
    output_k: int,
) -> Path:
    records = []
    seen: set[int] = set()
    with Path(predictions_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = int(row["qid"])
            if qid in seen:
                raise ValueError(f"Duplicate qid in predictions: {qid}")
            seen.add(qid)
            aids = list(dict.fromkeys(map(int, row["retrieved_aids"][:output_k])))
            records.append({"qid": qid, "relevant_laws": aids})
    if seen != expected_qids:
        missing = sorted(expected_qids - seen)
        extra = sorted(seen - expected_qids)
        raise ValueError(f"Submission qid mismatch; missing={missing[:10]}, extra={extra[:10]}")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("results.json", payload)
    return destination
