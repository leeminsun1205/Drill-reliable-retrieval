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
        rm3_enabled: bool = False,
        rm3_feedback_documents: int = 5,
        rm3_feedback_terms: int = 10,
        rm3_original_query_weight: float = 0.7,
        rm3_dirichlet_mu: float = 1000.0,
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
        if rm3_feedback_documents <= 0 or rm3_feedback_terms <= 0:
            raise ValueError("RM3 feedback documents and terms must be positive")
        if not 0 <= rm3_original_query_weight <= 1:
            raise ValueError("RM3 original_query_weight must be in [0, 1]")
        if rm3_dirichlet_mu <= 0:
            raise ValueError("RM3 dirichlet_mu must be positive")
        self.k1 = k1
        self.b = b
        self.variant = variant
        self.delta = delta
        self.text_processor = text_processor or build_token_processor()
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency = max_document_frequency
        self.max_document_frequency_ratio = max_document_frequency_ratio
        self.rm3_enabled = rm3_enabled
        self.rm3_feedback_documents = rm3_feedback_documents
        self.rm3_feedback_terms = rm3_feedback_terms
        self.rm3_original_query_weight = rm3_original_query_weight
        self.rm3_dirichlet_mu = rm3_dirichlet_mu
        self._aids: list[int] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._vocabulary_stats: dict[str, int | float | None] = {}
        self._doc_term_counts: list[Counter[str]] = []
        self._collection_probabilities: dict[str, float] = {}
        self._last_feedback: list[dict[str, object]] = []

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
        self._doc_term_counts = []
        self._collection_probabilities = {}
        if self.rm3_enabled:
            self._doc_term_counts = [Counter() for _ in range(document_count)]
            collection_frequency: Counter[str] = Counter()
            for term, items in retained.items():
                for position, frequency in items:
                    self._doc_term_counts[position][term] = frequency
                    collection_frequency[term] += frequency
            collection_length = sum(collection_frequency.values())
            self._collection_probabilities = {
                term: frequency / collection_length
                for term, frequency in collection_frequency.items()
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

    def _score(self, term_weights: dict[str, float], top_k: int) -> list[tuple[int, float]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for term, query_weight in term_weights.items():
            idf = self._idf.get(term)
            if idf is None or query_weight <= 0:
                continue
            for position, frequency in self._postings[term]:
                length_norm = 1 - self.b + self.b * self._doc_lengths[position] / self._avgdl
                tf_score = (frequency * (self.k1 + 1)) / (
                    frequency + self.k1 * length_norm
                )
                if self.variant == "bm25_plus":
                    tf_score += self.delta
                scores[position] += query_weight * idf * tf_score
        best = heapq.nlargest(
            min(top_k, len(scores)),
            scores.items(),
            key=lambda item: (item[1], -self._aids[item[0]]),
        )
        if len(best) < min(top_k, len(self._aids)):
            selected = {position for position, _ in best}
            zero_score_positions = sorted(
                (position for position in range(len(self._aids)) if position not in selected),
                key=lambda position: self._aids[position],
            )
            needed = min(top_k, len(self._aids)) - len(best)
            best.extend((position, 0.0) for position in zero_score_positions[:needed])
        return best

    def _rm3_query_model(
        self, query_tokens: list[str], first_pass: list[tuple[int, float]]
    ) -> tuple[dict[str, float], dict[str, object]]:
        query_counts = Counter(term for term in query_tokens if term in self._idf)
        if not query_counts:
            return {}, {"feedback_aids": [], "expanded_terms": []}
        positive = [item for item in first_pass if item[1] > 0][
            : self.rm3_feedback_documents
        ]
        if not positive:
            weights = {term: count / sum(query_counts.values()) for term, count in query_counts.items()}
            return weights, {"feedback_aids": [], "expanded_terms": []}

        log_likelihoods = []
        for position, _ in positive:
            length = self._doc_lengths[position]
            counts = self._doc_term_counts[position]
            value = 0.0
            for term, query_frequency in query_counts.items():
                collection_probability = self._collection_probabilities[term]
                probability = (
                    counts.get(term, 0)
                    + self.rm3_dirichlet_mu * collection_probability
                ) / (length + self.rm3_dirichlet_mu)
                value += query_frequency * math.log(probability)
            log_likelihoods.append(value)
        maximum = max(log_likelihoods)
        likelihoods = [math.exp(value - maximum) for value in log_likelihoods]
        likelihood_total = sum(likelihoods)
        document_weights = [value / likelihood_total for value in likelihoods]

        feedback_model: defaultdict[str, float] = defaultdict(float)
        for (position, _), document_weight in zip(positive, document_weights):
            length = self._doc_lengths[position]
            for term, frequency in self._doc_term_counts[position].items():
                probability = (
                    frequency
                    + self.rm3_dirichlet_mu * self._collection_probabilities[term]
                ) / (length + self.rm3_dirichlet_mu)
                feedback_model[term] += document_weight * probability
        selected_feedback = sorted(
            feedback_model.items(), key=lambda item: (-item[1], item[0])
        )[: self.rm3_feedback_terms]
        selected_total = sum(weight for _, weight in selected_feedback)
        original_total = sum(query_counts.values())
        query_model = {
            term: self.rm3_original_query_weight * count / original_total
            for term, count in query_counts.items()
        }
        if selected_total > 0:
            for term, weight in selected_feedback:
                query_model[term] = query_model.get(term, 0.0) + (
                    (1 - self.rm3_original_query_weight) * weight / selected_total
                )
        audit = {
            "feedback_aids": [self._aids[position] for position, _ in positive],
            "expanded_terms": [
                {"term": term, "weight": float(query_model[term])}
                for term, _ in selected_feedback
                if term not in query_counts
            ],
        }
        return query_model, audit

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if not self._aids:
            raise RuntimeError("fit() must be called before retrieve()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        results: list[RetrievalResult] = []
        self._last_feedback = []
        for query in queries:
            query_tokens = self.text_processor(query.question)
            base_weights = {term: 1.0 for term in set(query_tokens)}
            if self.rm3_enabled:
                first_pass = self._score(
                    base_weights, max(top_k, self.rm3_feedback_documents)
                )
                term_weights, audit = self._rm3_query_model(query_tokens, first_pass)
                best = self._score(term_weights, top_k)
                self._last_feedback.append({"qid": query.qid, **audit})
            else:
                best = self._score(base_weights, top_k)
            results.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(self._aids[position] for position, _ in best),
                    scores=tuple(float(score) for _, score in best),
                )
            )
        return results

    def feedback_records(self) -> list[dict[str, object]]:
        return list(self._last_feedback)

    def run_metadata(self) -> dict[str, object]:
        metadata = getattr(self.text_processor, "metadata", None)
        return {
            "variant": self.variant,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta if self.variant == "bm25_plus" else None,
            "preprocessing": metadata() if callable(metadata) else None,
            "vocabulary": self._vocabulary_stats,
            "feedback": {
                "enabled": self.rm3_enabled,
                "method": "rm3" if self.rm3_enabled else "none",
                "feedback_documents": self.rm3_feedback_documents,
                "feedback_terms": self.rm3_feedback_terms,
                "original_query_weight": self.rm3_original_query_weight,
                "dirichlet_mu": self.rm3_dirichlet_mu,
            },
        }
