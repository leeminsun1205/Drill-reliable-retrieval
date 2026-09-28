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
