"""Dependency-light BM25 using a deterministic Unicode tokenization baseline."""

from collections import Counter, defaultdict
import heapq
import math
from typing import Sequence

from drill_ir.data.models import Article, Query, RetrievalResult
from drill_ir.preprocessing import TokenProcessor, build_token_processor

def tokenize(text: str) -> list[str]:
    """Backward-compatible default tokenizer used by the original baseline."""
    return build_token_processor()(text)


class BM25Retriever:
    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        variant: str = "bm25",
        delta: float = 1.0,
        text_processor: TokenProcessor | None = None,
        min_document_frequency: int = 1,
        max_document_frequency: int | None = None,
        max_document_frequency_ratio: float = 1.0,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if variant not in {"bm25", "bm25_plus"}:
            raise ValueError("BM25 variant must be 'bm25' or 'bm25_plus'")
        if delta < 0:
            raise ValueError("BM25+ delta must be non-negative")
        if min_document_frequency <= 0:
            raise ValueError("BM25 min_document_frequency must be positive")
        if max_document_frequency is not None and max_document_frequency <= 0:
            raise ValueError("BM25 max_document_frequency must be null or positive")
        if not 0 < max_document_frequency_ratio <= 1:
            raise ValueError("BM25 max_document_frequency_ratio must be in (0, 1]")
        self.k1 = k1
        self.b = b
        self.variant = variant
        self.delta = delta
        self.text_processor = text_processor or build_token_processor()
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency = max_document_frequency
        self.max_document_frequency_ratio = max_document_frequency_ratio
        self._aids: list[int] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._vocabulary_stats: dict[str, int | float | None] = {}

    def fit(self, articles: Sequence[Article]) -> None:
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        self._aids = []
        self._doc_lengths = []
        for position, article in enumerate(articles):
            counts = Counter(self.text_processor(article.text))
            self._aids.append(article.aid)
            self._doc_lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((position, frequency))

        document_count = len(self._aids)
        if not document_count:
            raise ValueError("Cannot fit BM25 on an empty corpus")
        maximum_df = int(document_count * self.max_document_frequency_ratio)
        if self.max_document_frequency is not None:
            maximum_df = min(maximum_df, self.max_document_frequency)
        retained = {
            term: items
            for term, items in postings.items()
            if self.min_document_frequency <= len(items) <= maximum_df
        }
        self._postings = retained
        # Length normalization must use the same retained vocabulary as scoring.
        self._doc_lengths = [0] * document_count
        for items in retained.values():
            for position, frequency in items:
                self._doc_lengths[position] += frequency
        self._avgdl = sum(self._doc_lengths) / document_count
        self._idf = {
            term: math.log(1.0 + (document_count - len(items) + 0.5) / (len(items) + 0.5))
            for term, items in self._postings.items()
        }
        self._vocabulary_stats = {
            "original_terms": len(postings),
            "retained_terms": len(retained),
            "pruned_terms": len(postings) - len(retained),
            "min_document_frequency": self.min_document_frequency,
            "max_document_frequency": self.max_document_frequency,
            "max_document_frequency_ratio": self.max_document_frequency_ratio,
            "effective_max_document_frequency": maximum_df,
        }

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if not self._aids:
            raise RuntimeError("fit() must be called before retrieve()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        results: list[RetrievalResult] = []
        for query in queries:
            scores: defaultdict[int, float] = defaultdict(float)
            for term in set(self.text_processor(query.question)):
                idf = self._idf.get(term)
                if idf is None:
                    continue
                for position, frequency in self._postings[term]:
                    length_norm = 1 - self.b + self.b * self._doc_lengths[position] / self._avgdl
                    tf_score = (frequency * (self.k1 + 1)) / (
                        frequency + self.k1 * length_norm
                    )
                    if self.variant == "bm25_plus":
                        tf_score += self.delta
                    scores[position] += idf * tf_score
            best = heapq.nlargest(
                min(top_k, len(scores)), scores.items(), key=lambda item: (item[1], -self._aids[item[0]])
            )
            if len(best) < min(top_k, len(self._aids)):
                selected = {position for position, _ in best}
                zero_score_positions = sorted(
                    (position for position in range(len(self._aids)) if position not in selected),
                    key=lambda position: self._aids[position],
                )
                needed = min(top_k, len(self._aids)) - len(best)
                best.extend((position, 0.0) for position in zero_score_positions[:needed])
            results.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(self._aids[position] for position, _ in best),
                    scores=tuple(float(score) for _, score in best),
                )
            )
        return results

    def run_metadata(self) -> dict[str, object]:
        metadata = getattr(self.text_processor, "metadata", None)
        return {
            "variant": self.variant,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta if self.variant == "bm25_plus" else None,
            "preprocessing": metadata() if callable(metadata) else None,
            "vocabulary": self._vocabulary_stats,
        }
