"""Validation and packaging of official Codabench submissions."""

import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from drill_ir.evaluation import select_by_score_gap


def _write_submission(records: list[dict[str, object]], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("results.json", payload)
    return destination


def _validate_qids(seen: set[int], expected_qids: set[int]) -> None:
    if seen != expected_qids:
        missing = sorted(expected_qids - seen)
        extra = sorted(seen - expected_qids)
        raise ValueError(
            f"Submission qid mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )


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
    _validate_qids(seen, expected_qids)
    return _write_submission(records, destination)


def package_threshold_submission(
    predictions_path: str | Path,
    expected_qids: set[int],
    destination: str | Path,
    *,
    score_threshold: float,
    fallback_k: int,
) -> Path:
    """Package all scores at or above a threshold, with empty-only top-k fallback."""
    if not math.isfinite(score_threshold):
        raise ValueError("score_threshold must be finite")
    if isinstance(fallback_k, bool) or not isinstance(fallback_k, int) or fallback_k <= 0:
        raise ValueError("fallback_k must be a positive integer")

    records = []
    seen: set[int] = set()
    with Path(predictions_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = int(row["qid"])
            if qid in seen:
                raise ValueError(f"Duplicate qid in predictions: {qid}")
            seen.add(qid)
            aids = tuple(map(int, row["retrieved_aids"]))
            scores = tuple(map(float, row["scores"]))
            if len(aids) != len(scores):
                raise ValueError(f"Ranking aids/scores length mismatch for qid={qid}")
            if not aids:
                raise ValueError(f"Threshold submission requires candidates for qid={qid}")
            if any(not math.isfinite(score) for score in scores):
                raise ValueError(f"Ranking contains non-finite scores for qid={qid}")
            if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
                raise ValueError(f"Ranking scores must be descending for qid={qid}")
            selected_count = sum(score >= score_threshold for score in scores)
            if selected_count == 0:
                selected_count = min(fallback_k, len(aids))
            selected = list(dict.fromkeys(aids[:selected_count]))
            records.append({"qid": qid, "relevant_laws": selected})

    _validate_qids(seen, expected_qids)
    return _write_submission(records, destination)


def package_score_gap_submission(
    predictions_path: str | Path,
    expected_qids: set[int],
    destination: str | Path,
    *,
    minimum_gap: float,
    fallback_k: int,
    maximum_k: int,
) -> Path:
    """Package query-adaptive prefixes selected by the strongest early gap."""
    rankings: dict[int, tuple[list[int], list[float]]] = {}
    with Path(predictions_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = int(row["qid"])
            if qid in rankings:
                raise ValueError(f"Duplicate qid in predictions: {qid}")
            rankings[qid] = (
                list(map(int, row["retrieved_aids"])),
                list(map(float, row["scores"])),
            )
    _validate_qids(set(rankings), expected_qids)
    selected = select_by_score_gap(
        rankings,
        minimum_gap=minimum_gap,
        fallback_k=fallback_k,
        maximum_k=maximum_k,
    )
    records = [
        {"qid": qid, "relevant_laws": selected[qid]} for qid in sorted(selected)
    ]
    return _write_submission(records, destination)
