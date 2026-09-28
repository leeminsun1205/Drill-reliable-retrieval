"""Official DRiLL set-based macro metrics."""

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class DrillMetrics:
    precision: float
    recall: float
    f2: float
    queries: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_predictions(
    gold: Mapping[int, Iterable[int]], predictions: Mapping[int, Iterable[int]]
) -> DrillMetrics:
    """Macro-average P/R per query, then derive F2 from macro P/R."""
    if not gold:
        raise ValueError("Cannot evaluate an empty gold mapping")
    extra = set(predictions) - set(gold)
    if extra:
        raise ValueError(f"Predictions contain unknown qids: {sorted(extra)[:10]}")

    precisions: list[float] = []
    recalls: list[float] = []
    for qid, relevant in gold.items():
        gold_set = set(relevant)
        pred_set = set(predictions.get(qid, ()))
        if not gold_set:
            raise ValueError(f"Gold labels are empty for qid={qid}")
        correct = len(gold_set & pred_set)
        precisions.append(correct / len(pred_set) if pred_set else 0.0)
        recalls.append(correct / len(gold_set))

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    denominator = 4 * precision + recall
    f2 = 5 * precision * recall / denominator if denominator else 0.0
    return DrillMetrics(precision=precision, recall=recall, f2=f2, queries=len(gold))


def macro_recall_at_k(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Iterable[int]],
    cutoffs: Iterable[int],
) -> dict[int, float]:
    """Calculate macro Recall@k for ranked candidate lists.

    This is a diagnostic candidate-retrieval metric, not a replacement for the
    official DRiLL set-based metrics. Missing rankings contribute zero recall.
    """
    if not gold:
        raise ValueError("Cannot evaluate an empty gold mapping")
    extra = set(rankings) - set(gold)
    if extra:
        raise ValueError(f"Rankings contain unknown qids: {sorted(extra)[:10]}")

    cutoff_values = list(cutoffs)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cutoff_values):
        raise ValueError("Recall@k cutoffs must be integers")
    normalized_cutoffs = sorted(set(cutoff_values))
    if not normalized_cutoffs:
        raise ValueError("At least one Recall@k cutoff is required")
    if normalized_cutoffs[0] <= 0:
        raise ValueError("Recall@k cutoffs must be positive integers")

    totals = {cutoff: 0.0 for cutoff in normalized_cutoffs}
    for qid, relevant in gold.items():
        gold_set = set(relevant)
        if not gold_set:
            raise ValueError(f"Gold labels are empty for qid={qid}")
        ranked_aids = list(rankings.get(qid, ()))
        for cutoff in normalized_cutoffs:
            retrieved = set(ranked_aids[:cutoff])
            totals[cutoff] += len(gold_set & retrieved) / len(gold_set)

    query_count = len(gold)
    return {cutoff: totals[cutoff] / query_count for cutoff in normalized_cutoffs}
