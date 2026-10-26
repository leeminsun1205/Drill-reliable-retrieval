"""Official DRiLL set-based macro metrics."""

from dataclasses import asdict, dataclass
import math
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


def select_by_score_gap(
    rankings: Mapping[int, tuple[Iterable[int], Iterable[float]]],
    *,
    minimum_gap: float,
    fallback_k: int,
    maximum_k: int,
) -> dict[int, list[int]]:
    """Select a query-specific prefix at its strongest early score drop.

    Only gaps after ranks 1 through ``maximum_k - 1`` are considered. If no
    gap reaches ``minimum_gap``, the deterministic fixed ``fallback_k`` prefix
    is used. This is a final-set policy; it never changes candidate ordering.
    """
    if not math.isfinite(minimum_gap) or minimum_gap < 0:
        raise ValueError("minimum_gap must be finite and non-negative")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (fallback_k, maximum_k)
    ):
        raise ValueError("fallback_k and maximum_k must be positive integers")
    if fallback_k > maximum_k:
        raise ValueError("fallback_k cannot exceed maximum_k")

    predictions: dict[int, list[int]] = {}
    for qid, (aids_iterable, scores_iterable) in rankings.items():
        aids = tuple(map(int, aids_iterable))
        scores = tuple(map(float, scores_iterable))
        if len(aids) != len(scores):
            raise ValueError(f"Ranking aids/scores length mismatch for qid={qid}")
        if not aids:
            raise ValueError(f"Score-gap selection requires candidates for qid={qid}")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError(f"Ranking contains non-finite scores for qid={qid}")
        if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
            raise ValueError(f"Ranking scores must be descending for qid={qid}")

        gap_count = min(maximum_k - 1, len(scores) - 1)
        gaps = [scores[index] - scores[index + 1] for index in range(gap_count)]
        if gaps and max(gaps) >= minimum_gap:
            # Earlier boundary wins an exact tie, yielding the smaller set.
            selected_count = gaps.index(max(gaps)) + 1
        else:
            selected_count = min(fallback_k, maximum_k, len(aids))
        predictions[int(qid)] = list(dict.fromkeys(aids[:selected_count]))
    return predictions


def tune_score_gap(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, tuple[Iterable[int], Iterable[float]]],
    *,
    fallback_cutoffs: Iterable[int] = (1, 2, 3),
    maximum_cutoffs: Iterable[int] = (3, 4, 5),
) -> dict[str, object]:
    """Tune score-gap prefix selection on labeled data.

    Candidate gap thresholds are exact observed early gaps plus a value just
    above the maximum (the fixed-fallback case), avoiding an arbitrary grid.
    """
    if set(rankings) != set(gold):
        raise ValueError("Score-gap rankings must contain every gold qid exactly once")
    fallbacks = sorted(set(fallback_cutoffs))
    maxima = sorted(set(maximum_cutoffs))
    if not fallbacks or not maxima:
        raise ValueError("Score-gap tuning requires fallback and maximum cutoffs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in fallbacks + maxima):
        raise ValueError("Score-gap cutoffs must be positive integers")

    candidates = {0.0}
    for _, scores_iterable in rankings.values():
        scores = tuple(map(float, scores_iterable))
        for index in range(min(max(maxima) - 1, len(scores) - 1)):
            candidates.add(scores[index] - scores[index + 1])
    candidates.add(math.nextafter(max(candidates), math.inf))

    best: dict[str, object] | None = None
    evaluated = 0
    for maximum_k in maxima:
        for fallback_k in fallbacks:
            if fallback_k > maximum_k:
                continue
            for minimum_gap in sorted(candidates):
                predictions = select_by_score_gap(
                    rankings,
                    minimum_gap=minimum_gap,
                    fallback_k=fallback_k,
                    maximum_k=maximum_k,
                )
                metrics = evaluate_predictions(gold, predictions)
                sizes = [len(values) for values in predictions.values()]
                candidate: dict[str, object] = {
                    "minimum_gap": minimum_gap,
                    "fallback_k": fallback_k,
                    "maximum_k": maximum_k,
                    **metrics.to_dict(),
                    "output_size": {
                        "minimum": min(sizes),
                        "median": median(sizes),
                        "mean": sum(sizes) / len(sizes),
                        "maximum": max(sizes),
                    },
                }
                evaluated += 1
                key = (
                    candidate["f2"], candidate["precision"], candidate["recall"],
                    -candidate["output_size"]["mean"], -minimum_gap,
                )
                if best is None:
                    best = candidate
                else:
                    best_key = (
                        best["f2"], best["precision"], best["recall"],
                        -best["output_size"]["mean"], -best["minimum_gap"],
                    )
                    if key > best_key:
                        best = candidate
    assert best is not None
    return {
        "selection": "largest_early_score_gap_with_fixed_fallback",
        "tuned_on": "development_split",
        "evaluated_policies": evaluated,
        "best": best,
    }


def evaluate_oracle_prefix(
    gold: Mapping[int, Iterable[int]],
    rankings: Mapping[int, tuple[Iterable[int], Iterable[float]]],
    *,
    maximum_k: int = 5,
) -> dict[str, object]:
    """Measure optimistic headroom from choosing the best prefix per query."""
    predictions: dict[int, list[int]] = {}
    selected_sizes: list[int] = []
    for qid, relevant in gold.items():
        aids = list(rankings[qid][0])
        gold_set = set(relevant)
        best_key: tuple[float, float, float, int] | None = None
        best_k = 1
        for cutoff in range(1, min(maximum_k, len(aids)) + 1):
            selected = set(aids[:cutoff])
            precision = len(selected & gold_set) / cutoff
            recall = len(selected & gold_set) / len(gold_set)
            denominator = 4 * precision + recall
            f2 = 5 * precision * recall / denominator if denominator else 0.0
            key = (f2, precision, recall, -cutoff)
            if best_key is None or key > best_key:
                best_key, best_k = key, cutoff
        predictions[qid] = aids[:best_k]
        selected_sizes.append(best_k)
    return {
        "label": "oracle_upper_bound_not_deployable",
        "maximum_k": maximum_k,
        **evaluate_predictions(gold, predictions).to_dict(),
        "output_size": {
            "minimum": min(selected_sizes),
            "median": median(selected_sizes),
            "mean": sum(selected_sizes) / len(selected_sizes),
            "maximum": max(selected_sizes),
        },
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
