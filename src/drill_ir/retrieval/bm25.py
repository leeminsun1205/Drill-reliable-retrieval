"""Dependency-light BM25 using a deterministic Unicode tokenization baseline."""

from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from drill_ir.data.models import Article, Query, RetrievalResult
from drill_ir.preprocessing import TokenProcessor, build_token_processor


class BM25IndexCache:
    """Validated, non-pickle on-disk cache for a fitted BM25 index."""

    VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, corpus_fingerprint: str, index_key: str) -> Path:
        digest = hashlib.sha256(
            f"{corpus_fingerprint}\0{index_key}".encode()
        ).hexdigest()[:20]
        return self.root / digest

    def load(self, *, corpus_fingerprint: str, index_key: str) -> dict[str, object] | None:
        directory = self._directory(corpus_fingerprint, index_key)
        required = {
            "metadata": directory / "metadata.json",
            "terms": directory / "terms.json",
            "aids": directory / "aids.npy",
            "doc_lengths": directory / "doc_lengths.npy",
            "offsets": directory / "postings_offsets.npy",
            "positions": directory / "postings_positions.npy",
            "frequencies": directory / "postings_frequencies.npy",
        }
        if not all(path.is_file() for path in required.values()):
            return None
        metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("version") != self.VERSION
            or metadata.get("corpus_fingerprint") != corpus_fingerprint
            or metadata.get("index_key") != index_key
        ):
            return None
        terms = json.loads(required["terms"].read_text(encoding="utf-8"))
        aids = np.load(required["aids"], allow_pickle=False)
        doc_lengths = np.load(required["doc_lengths"], allow_pickle=False)
        offsets = np.load(required["offsets"], allow_pickle=False)
        positions = np.load(required["positions"], allow_pickle=False)
        frequencies = np.load(required["frequencies"], allow_pickle=False)
        if (
            not isinstance(terms, list)
            or len(offsets) != len(terms) + 1
            or len(positions) != len(frequencies)
            or int(offsets[-1]) != len(positions)
            or len(aids) != len(doc_lengths)
        ):
            raise ValueError(f"Invalid BM25 cache: {directory}")
        postings = {
            str(term): list(
                zip(
                    positions[int(offsets[index]) : int(offsets[index + 1])].astype(int).tolist(),
                    frequencies[int(offsets[index]) : int(offsets[index + 1])].astype(int).tolist(),
                )
            )
            for index, term in enumerate(terms)
        }
        return {
            "aids": aids.astype(int).tolist(),
            "doc_lengths": doc_lengths.astype(int).tolist(),
            "postings": postings,
            "avgdl": float(metadata["avgdl"]),
            "vocabulary_stats": metadata["vocabulary_stats"],
            "directory": directory,
        }

    def save(
        self,
        *,
        corpus_fingerprint: str,
        index_key: str,
        aids: Sequence[int],
        doc_lengths: Sequence[int],
        postings: dict[str, list[tuple[int, int]]],
        avgdl: float,
        vocabulary_stats: dict[str, int | float | None],
    ) -> Path:
        directory = self._directory(corpus_fingerprint, index_key)
        directory.mkdir(parents=True, exist_ok=True)
        terms = sorted(postings)
        offsets = [0]
        positions: list[int] = []
        frequencies: list[int] = []
        for term in terms:
            for position, frequency in postings[term]:
                positions.append(position)
                frequencies.append(frequency)
            offsets.append(len(positions))
        np.save(directory / "aids.npy", np.asarray(aids, dtype=np.int64), allow_pickle=False)
        np.save(
            directory / "doc_lengths.npy",
            np.asarray(doc_lengths, dtype=np.int64),
            allow_pickle=False,
        )
        np.save(
            directory / "postings_offsets.npy",
            np.asarray(offsets, dtype=np.int64),
            allow_pickle=False,
        )
        np.save(
            directory / "postings_positions.npy",
            np.asarray(positions, dtype=np.int64),
            allow_pickle=False,
        )
        np.save(
            directory / "postings_frequencies.npy",
            np.asarray(frequencies, dtype=np.int64),
            allow_pickle=False,
        )
        (directory / "terms.json").write_text(
            json.dumps(terms, ensure_ascii=False), encoding="utf-8"
        )
        metadata = {
            "version": self.VERSION,
            "corpus_fingerprint": corpus_fingerprint,
            "index_key": index_key,
            "avgdl": avgdl,
            "vocabulary_stats": vocabulary_stats,
            "documents": len(aids),
            "terms": len(terms),
            "postings": len(positions),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory


class BM25SIndexCache:
    """Validated cache around bm25s' non-pickle sparse index format."""

    VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, corpus_fingerprint: str, index_key: str) -> Path:
        digest = hashlib.sha256(
            f"bm25s\0{corpus_fingerprint}\0{index_key}".encode()
        ).hexdigest()[:20]
        return self.root / digest

    def load(
        self, *, corpus_fingerprint: str, index_key: str
    ) -> tuple[Any, list[int], dict[str, int | float | None], Path] | None:
        directory = self._directory(corpus_fingerprint, index_key)
        metadata_path = directory / "metadata.json"
        aids_path = directory / "aids.npy"
        index_directory = directory / "bm25s_index"
        if not metadata_path.is_file() or not aids_path.is_file() or not index_directory.is_dir():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("version") != self.VERSION
            or metadata.get("corpus_fingerprint") != corpus_fingerprint
            or metadata.get("index_key") != index_key
        ):
            return None
        aids = np.load(aids_path, allow_pickle=False)
        if aids.ndim != 1 or len(aids) != int(metadata.get("documents", -1)):
            raise ValueError(f"Invalid BM25S cache: {directory}")
        try:
            import bm25s
        except ImportError as error:
            raise RuntimeError(
                "BM25S retrieval requires the sparse dependency. "
                "Install with: python -m pip install -e '.[sparse]'"
            ) from error
        retriever = bm25s.BM25.load(index_directory, load_corpus=False, mmap=True)
        return (
            retriever,
            aids.astype(int).tolist(),
            dict(metadata.get("vocabulary_stats", {})),
            directory,
        )

    def save(
        self,
        retriever: Any,
        *,
        corpus_fingerprint: str,
        index_key: str,
        aids: Sequence[int],
        vocabulary_stats: dict[str, int | float | None],
    ) -> Path:
        directory = self._directory(corpus_fingerprint, index_key)
        directory.mkdir(parents=True, exist_ok=True)
        retriever.save(directory / "bm25s_index")
        np.save(directory / "aids.npy", np.asarray(aids, dtype=np.int64), allow_pickle=False)
        metadata = {
            "version": self.VERSION,
            "backend": "bm25s",
            "corpus_fingerprint": corpus_fingerprint,
            "index_key": index_key,
            "documents": len(aids),
            "vocabulary_stats": vocabulary_stats,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

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
        cache: BM25IndexCache | None = None,
        require_cache_hit: bool = False,
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
        self.cache = cache
        self.require_cache_hit = require_cache_hit
        if self.require_cache_hit and self.cache is None:
            raise ValueError("require_cache_hit requires an enabled BM25 cache")
        self.cache_hit = False
        self.cache_directory: str | None = None
        self.corpus_fingerprint: str | None = None
        self._aids: list[int] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._vocabulary_stats: dict[str, int | float | None] = {}
        self._doc_term_counts: list[Counter[str]] = []
        self._collection_probabilities: dict[str, float] = {}
        self._last_feedback: list[dict[str, object]] = []

    @staticmethod
    def _corpus_fingerprint(articles: Sequence[Article]) -> str:
        digest = hashlib.sha256()
        for article in articles:
            digest.update(str(article.aid).encode())
            digest.update(b"\0")
            digest.update(article.law_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(article.text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    @property
    def cache_key(self) -> str:
        metadata = getattr(self.text_processor, "metadata", None)
        payload = {
            "preprocessing": metadata() if callable(metadata) else None,
            "min_document_frequency": self.min_document_frequency,
            "max_document_frequency": self.max_document_frequency,
            "max_document_frequency_ratio": self.max_document_frequency_ratio,
            "rm3_enabled": self.rm3_enabled,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _restore_cache(self, payload: dict[str, object]) -> None:
        self._aids = list(payload["aids"])  # type: ignore[arg-type]
        self._doc_lengths = list(payload["doc_lengths"])  # type: ignore[arg-type]
        self._postings = dict(payload["postings"])  # type: ignore[arg-type]
        self._avgdl = float(payload["avgdl"])
        self._vocabulary_stats = dict(payload["vocabulary_stats"])  # type: ignore[arg-type]
        document_count = len(self._aids)
        self._idf = {
            term: math.log(1.0 + (document_count - len(items) + 0.5) / (len(items) + 0.5))
            for term, items in self._postings.items()
        }
        self._prepare_rm3_statistics(document_count)

    def _prepare_rm3_statistics(self, document_count: int) -> None:
        self._doc_term_counts = []
        self._collection_probabilities = {}
        if not self.rm3_enabled:
            return
        self._doc_term_counts = [Counter() for _ in range(document_count)]
        collection_frequency: Counter[str] = Counter()
        for term, items in self._postings.items():
            for position, frequency in items:
                self._doc_term_counts[position][term] = frequency
                collection_frequency[term] += frequency
        collection_length = sum(collection_frequency.values())
        self._collection_probabilities = {
            term: frequency / collection_length
            for term, frequency in collection_frequency.items()
        }

    def fit(self, articles: Sequence[Article]) -> None:
        if not articles:
            raise ValueError("Cannot fit BM25 on an empty corpus")
        self.corpus_fingerprint = self._corpus_fingerprint(articles)
        self.cache_hit = False
        self.cache_directory = None
        if self.cache is not None:
            cached = self.cache.load(
                corpus_fingerprint=self.corpus_fingerprint,
                index_key=self.cache_key,
            )
            if cached is not None:
                self._restore_cache(cached)
                self.cache_hit = True
                self.cache_directory = str(cached["directory"])
                print("[bm25] Index cache hit", flush=True)
                return
            print("[bm25] Index cache miss", flush=True)
            if self.require_cache_hit:
                raise RuntimeError("Required BM25 index cache is missing or invalid")
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
        self._prepare_rm3_statistics(document_count)
        self._vocabulary_stats = {
            "original_terms": len(postings),
            "retained_terms": len(retained),
            "pruned_terms": len(postings) - len(retained),
            "min_document_frequency": self.min_document_frequency,
            "max_document_frequency": self.max_document_frequency,
            "max_document_frequency_ratio": self.max_document_frequency_ratio,
            "effective_max_document_frequency": maximum_df,
        }
        if self.cache is not None:
            directory = self.cache.save(
                corpus_fingerprint=self.corpus_fingerprint,
                index_key=self.cache_key,
                aids=self._aids,
                doc_lengths=self._doc_lengths,
                postings=self._postings,
                avgdl=self._avgdl,
                vocabulary_stats=self._vocabulary_stats,
            )
            self.cache_directory = str(directory)
            print("[bm25] Saved index cache", flush=True)

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
            "backend": "native",
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
            "cache": {
                "enabled": self.cache is not None,
                "hit": self.cache_hit,
                "required": self.require_cache_hit,
                "directory": self.cache_directory,
                "corpus_fingerprint": self.corpus_fingerprint,
            },
        }


class BM25SRetriever:
    """bm25s sparse-matrix backend with project-controlled preprocessing.

    Standard BM25 uses bm25s' Lucene term-frequency and IDF definitions. The
    native implementation remains the reference oracle for ranking parity.
    """

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
        cache: BM25SIndexCache | None = None,
        require_cache_hit: bool = False,
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
        if require_cache_hit and cache is None:
            raise ValueError("require_cache_hit requires an enabled BM25S cache")
        self.k1 = k1
        self.b = b
        self.variant = variant
        self.delta = delta
        self.text_processor = text_processor or build_token_processor()
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency = max_document_frequency
        self.max_document_frequency_ratio = max_document_frequency_ratio
        self.cache = cache
        self.require_cache_hit = require_cache_hit
        self.cache_hit = False
        self.cache_directory: str | None = None
        self.corpus_fingerprint: str | None = None
        self._aids: list[int] = []
        self._retriever: Any = None
        self._vocabulary_stats: dict[str, int | float | None] = {}

    @staticmethod
    def _corpus_fingerprint(articles: Sequence[Article]) -> str:
        return BM25Retriever._corpus_fingerprint(articles)

    @property
    def cache_key(self) -> str:
        metadata = getattr(self.text_processor, "metadata", None)
        payload = {
            "backend": "bm25s",
            "bm25s_method": "lucene" if self.variant == "bm25" else "bm25+",
            "idf_method": "lucene",
            "dtype": "float64",
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta if self.variant == "bm25_plus" else None,
            "preprocessing": metadata() if callable(metadata) else None,
            "min_document_frequency": self.min_document_frequency,
            "max_document_frequency": self.max_document_frequency,
            "max_document_frequency_ratio": self.max_document_frequency_ratio,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _bm25s_module() -> Any:
        try:
            import bm25s
        except ImportError as error:
            raise RuntimeError(
                "BM25S retrieval requires the sparse dependency. "
                "Install with: python -m pip install -e '.[sparse]'"
            ) from error
        return bm25s

    def fit(self, articles: Sequence[Article]) -> None:
        if not articles:
            raise ValueError("Cannot fit BM25S on an empty corpus")
        self.corpus_fingerprint = self._corpus_fingerprint(articles)
        self.cache_hit = False
        self.cache_directory = None
        if self.cache is not None:
            cached = self.cache.load(
                corpus_fingerprint=self.corpus_fingerprint,
                index_key=self.cache_key,
            )
            if cached is not None:
                self._retriever, self._aids, self._vocabulary_stats, directory = cached
                self.cache_hit = True
                self.cache_directory = str(directory)
                print("[bm25s] Index cache hit", flush=True)
                return
            print("[bm25s] Index cache miss", flush=True)
            if self.require_cache_hit:
                raise RuntimeError("Required BM25S index cache is missing or invalid")

        tokenized = [self.text_processor(article.text) for article in articles]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        document_count = len(articles)
        maximum_df = int(document_count * self.max_document_frequency_ratio)
        if self.max_document_frequency is not None:
            maximum_df = min(maximum_df, self.max_document_frequency)
        retained = {
            term
            for term, frequency in document_frequency.items()
            if self.min_document_frequency <= frequency <= maximum_df
        }
        tokenized = [[term for term in tokens if term in retained] for tokens in tokenized]
        self._vocabulary_stats = {
            "original_terms": len(document_frequency),
            "retained_terms": len(retained),
            "pruned_terms": len(document_frequency) - len(retained),
            "min_document_frequency": self.min_document_frequency,
            "max_document_frequency": self.max_document_frequency,
            "max_document_frequency_ratio": self.max_document_frequency_ratio,
            "effective_max_document_frequency": maximum_df,
        }
        bm25s = self._bm25s_module()
        method = "lucene" if self.variant == "bm25" else "bm25+"
        self._retriever = bm25s.BM25(
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            method=method,
            idf_method="lucene",
            dtype="float64",
            backend="numpy",
        )
        self._retriever.index(tokenized, show_progress=True, leave_progress=False)
        self._aids = [article.aid for article in articles]
        if self.cache is not None:
            directory = self.cache.save(
                self._retriever,
                corpus_fingerprint=self.corpus_fingerprint,
                index_key=self.cache_key,
                aids=self._aids,
                vocabulary_stats=self._vocabulary_stats,
            )
            self.cache_directory = str(directory)
            print("[bm25s] Saved index cache", flush=True)

    def _resolve_candidates(
        self,
        row_positions: np.ndarray,
        row_scores: np.ndarray,
        *,
        result_k: int,
        searched_k: int,
    ) -> tuple[list[int], list[float]] | None:
        positive = row_scores > 0
        positive_positions = row_positions[positive]
        positive_scores = row_scores[positive]
        aid_array = np.asarray(self._aids, dtype=np.int64)
        order = np.lexsort((aid_array[positive_positions], -positive_scores))
        positive_positions = positive_positions[order]
        positive_scores = positive_scores[order]
        if len(positive_positions) < result_k:
            selected = set(positive_positions.astype(int).tolist())
            zeros = sorted(
                (position for position in range(len(self._aids)) if position not in selected),
                key=lambda position: self._aids[position],
            )[: result_k - len(positive_positions)]
            return (
                positive_positions.astype(int).tolist() + zeros,
                positive_scores.astype(float).tolist() + [0.0] * len(zeros),
            )
        boundary = float(positive_scores[result_k - 1])
        if searched_k >= len(self._aids) or float(row_scores[-1]) < boundary:
            return (
                positive_positions[:result_k].astype(int).tolist(),
                positive_scores[:result_k].astype(float).tolist(),
            )
        return None

    def _retrieve_one(self, query_tokens: list[str], top_k: int) -> tuple[list[int], list[float]]:
        result_k = min(top_k, len(self._aids))
        tokens = list(dict.fromkeys(query_tokens))
        depth = min(len(self._aids), max(result_k + 64, result_k * 2))
        while True:
            positions, scores = self._retriever.retrieve(
                [tokens], k=depth, sorted=True, show_progress=False, n_threads=0
            )
            resolved = self._resolve_candidates(
                np.asarray(positions[0], dtype=np.int64),
                np.asarray(scores[0], dtype=np.float64),
                result_k=result_k,
                searched_k=depth,
            )
            if resolved is not None:
                return resolved
            depth = min(len(self._aids), depth * 2)

    def retrieve(self, queries: Sequence[Query], top_k: int) -> list[RetrievalResult]:
        if self._retriever is None or not self._aids:
            raise RuntimeError("fit() must be called before retrieve()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not queries:
            return []
        result_k = min(top_k, len(self._aids))
        tokenized = [
            list(dict.fromkeys(self.text_processor(query.question))) for query in queries
        ]
        depth = min(len(self._aids), max(result_k + 64, result_k * 2))
        batch_positions, batch_scores = self._retriever.retrieve(
            tokenized, k=depth, sorted=True, show_progress=True, n_threads=0
        )
        results: list[RetrievalResult] = []
        for query, tokens, row_positions, row_scores in zip(
            queries, tokenized, batch_positions, batch_scores
        ):
            resolved = self._resolve_candidates(
                np.asarray(row_positions, dtype=np.int64),
                np.asarray(row_scores, dtype=np.float64),
                result_k=result_k,
                searched_k=depth,
            )
            positions, scores = (
                resolved if resolved is not None else self._retrieve_one(tokens, top_k)
            )
            results.append(
                RetrievalResult(
                    qid=query.qid,
                    aids=tuple(self._aids[position] for position in positions),
                    scores=tuple(scores),
                )
            )
        return results

    def feedback_records(self) -> list[dict[str, object]]:
        return []

    def run_metadata(self) -> dict[str, object]:
        metadata = getattr(self.text_processor, "metadata", None)
        bm25s = self._bm25s_module()
        return {
            "backend": "bm25s",
            "backend_version": getattr(bm25s, "__version__", "unknown"),
            "variant": self.variant,
            "bm25s_method": "lucene" if self.variant == "bm25" else "bm25+",
            "idf_method": "lucene",
            "dtype": "float64",
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta if self.variant == "bm25_plus" else None,
            "preprocessing": metadata() if callable(metadata) else None,
            "vocabulary": self._vocabulary_stats,
            "feedback": {"enabled": False, "method": "none"},
            "cache": {
                "enabled": self.cache is not None,
                "hit": self.cache_hit,
                "required": self.require_cache_hit,
                "directory": self.cache_directory,
                "corpus_fingerprint": self.corpus_fingerprint,
            },
        }
