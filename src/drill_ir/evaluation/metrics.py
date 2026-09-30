"""Official DRiLL set-based macro metrics."""

from dataclasses import asdict, dataclass
from statistics import median
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


def evaluate_fixed_cutoffs(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, Iterable[int]],
    cutoffs: Iterable[int],
) -> dict[int, DrillMetrics]:
    """Evaluate several final fixed-k policies from one saved ranking."""
    cutoff_values = list(cutoffs)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cutoff_values):
        raise ValueError("Output cutoffs must be integers")
    normalized_cutoffs = sorted(set(cutoff_values))
    if not normalized_cutoffs:
        raise ValueError("At least one output cutoff is required")
    if normalized_cutoffs[0] <= 0:
        raise ValueError("Output cutoffs must be positive integers")

    materialized_rankings = {qid: tuple(aids) for qid, aids in rankings.items()}
    return {
        cutoff: evaluate_predictions(
            gold,
            {qid: aids[:cutoff] for qid, aids in materialized_rankings.items()},
        )
        for cutoff in normalized_cutoffs
    }


def tune_score_thresholds(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, tuple[Iterable[int], Iterable[float]]],
    fallback_cutoffs: Iterable[int],
) -> dict[str, object]:
    """Find exact dev-optimal global score thresholds with non-empty fallbacks.

    A threshold retains every candidate whose score is greater than or equal to
    it. The fallback is used only when no candidate passes, matching the policy
    described by several DRiLL systems. This function tunes on labeled data; its
    result is not an unbiased estimate on the same queries.
    """
    fallback_values = sorted(set(fallback_cutoffs))
    if not fallback_values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in fallback_values
    ):
        raise ValueError("Fallback cutoffs must be positive integers")
    if set(rankings) != set(gold):
        raise ValueError("Threshold rankings must contain every gold qid exactly once")

    gold_sets = {qid: set(aids) for qid, aids in gold.items()}
    aids_by_qid: dict[int, tuple[int, ...]] = {}
    events: list[tuple[float, int]] = []
    for qid, (aids_iterable, scores_iterable) in rankings.items():
        aids = tuple(aids_iterable)
        scores = tuple(float(score) for score in scores_iterable)
        if len(aids) != len(scores):
            raise ValueError(f"Ranking aids/scores length mismatch for qid={qid}")
        if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
            raise ValueError(f"Ranking scores must be descending for qid={qid}")
        if not aids:
            raise ValueError(f"Threshold tuning requires candidates for qid={qid}")
        aids_by_qid[qid] = aids
        events.extend((score, qid) for score in scores)

    def query_values(qid: int, passed: int, fallback: int) -> tuple[float, float, int]:
        selected_count = passed if passed else min(fallback, len(aids_by_qid[qid]))
        selected = set(aids_by_qid[qid][:selected_count])
        correct = len(selected & gold_sets[qid])
        return (
            correct / selected_count if selected_count else 0.0,
            correct / len(gold_sets[qid]),
            selected_count,
        )

    query_count = len(gold_sets)
    results: dict[str, object] = {}
    unique_threshold_count = len({score for score, _ in events})
    for fallback in fallback_values:
        passed = {qid: 0 for qid in gold_sets}
        current = {
            qid: query_values(qid, 0, fallback)
            for qid in gold_sets
        }
        precision_sum = sum(value[0] for value in current.values())
        recall_sum = sum(value[1] for value in current.values())

        def snapshot(threshold: float | None) -> dict[str, object]:
            precision = precision_sum / query_count
            recall = recall_sum / query_count
            denominator = 4 * precision + recall
            sizes = [value[2] for value in current.values()]
            return {
                "threshold": threshold,
                "threshold_operator": ">=",
                "fallback_k": fallback,
                "precision": precision,
                "recall": recall,
                "f2": 5 * precision * recall / denominator if denominator else 0.0,
                "queries": query_count,
                "output_size": {
                    "minimum": min(sizes),
                    "median": median(sizes),
                    "mean": sum(sizes) / len(sizes),
                    "maximum": max(sizes),
                },
            }

        best = snapshot(None)
        event_index = 0
        sorted_events = sorted(events, reverse=True)
        while event_index < len(sorted_events):
            threshold = sorted_events[event_index][0]
            increments: dict[int, int] = {}
            while (
                event_index < len(sorted_events)
                and sorted_events[event_index][0] == threshold
            ):
                qid = sorted_events[event_index][1]
                increments[qid] = increments.get(qid, 0) + 1
                event_index += 1
            for qid, increment in increments.items():
                old_precision, old_recall, _ = current[qid]
                precision_sum -= old_precision
                recall_sum -= old_recall
                passed[qid] += increment
                current[qid] = query_values(qid, passed[qid], fallback)
                precision_sum += current[qid][0]
                recall_sum += current[qid][1]
            candidate = snapshot(threshold)
            candidate_key = (
                candidate["f2"],
                candidate["precision"],
                candidate["recall"],
                threshold,
            )
            best_threshold = best["threshold"]
            best_key = (
                best["f2"],
                best["precision"],
                best["recall"],
                float("inf") if best_threshold is None else best_threshold,
            )
            if candidate_key > best_key:
                best = candidate
        results[str(fallback)] = best

    return {
        "selection": "score_threshold_with_empty_only_fallback",
        "tuned_on": "development_split",
        "evaluated_unique_thresholds": unique_threshold_count,
        "best_by_fallback_k": results,
    }


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
