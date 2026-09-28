"""Dependency-light BM25 using a deterministic Unicode tokenization baseline."""

from collections import Counter, defaultdict
import heapq
import math
import re
import unicodedata
from typing import Sequence

from drill_ir.data.models import Article, Query, RetrievalResult

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).lower())


class BM25Retriever:
    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._aids: list[int] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def fit(self, articles: Sequence[Article]) -> None:
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        self._aids = []
        self._doc_lengths = []
        for position, article in enumerate(articles):
            counts = Counter(tokenize(article.text))
            self._aids.append(article.aid)
            self._doc_lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((position, frequency))

        document_count = len(self._aids)
        if not document_count:
            raise ValueError("Cannot fit BM25 on an empty corpus")
        self._avgdl = sum(self._doc_lengths) / document_count
        self._postings = dict(postings)
        self._idf = {
            term: math.log(1.0 + (document_count - len(items) + 0.5) / (len(items) + 0.5))
            for term, items in self._postings.items()
        }

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if not self._aids:
            raise RuntimeError("fit() must be called before retrieve()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        results: list[RetrievalResult] = []
        for query in queries:
            scores: defaultdict[int, float] = defaultdict(float)
            for term in set(tokenize(query.question)):
                idf = self._idf.get(term)
                if idf is None:
                    continue
                for position, frequency in self._postings[term]:
                    length_norm = 1 - self.b + self.b * self._doc_lengths[position] / self._avgdl
                    scores[position] += idf * (frequency * (self.k1 + 1)) / (
                        frequency + self.k1 * length_norm
                    )
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
