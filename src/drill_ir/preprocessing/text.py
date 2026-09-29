"""Composable normalization, Vietnamese word segmentation, and tokenization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, Protocol


WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")

STOPWORD_PRESETS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "vi_legal_conservative": frozenset(
        {
            "cả", "các", "cái", "chiếc", "chứ", "cứ", "cùng", "cũng",
            "đây", "để", "đó", "gì", "là", "lại", "lúc", "mà", "mỗi",
            "một_cách", "này", "ngay", "nhiều", "như", "nhưng", "những",
            "nữa", "qua", "ra", "rằng", "rất", "rồi", "so", "thì", "từng",
            "và", "vẫn", "vào", "vậy", "vì", "với", "vừa",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class TextNormalizerConfig:
    unicode_form: str | None = "NFKC"
    lowercase: bool = True
    normalize_whitespace: bool = True
    strip_control_characters: bool = True


class TextNormalizer:
    """Apply conservative transformations without changing the raw dataset."""

    def __init__(self, config: TextNormalizerConfig) -> None:
        if config.unicode_form not in {None, "NFC", "NFKC"}:
            raise ValueError("normalization.unicode_form must be null, 'NFC', or 'NFKC'")
        self.config = config

    def __call__(self, text: str) -> str:
        value = str(text)
        if self.config.unicode_form is not None:
            value = unicodedata.normalize(self.config.unicode_form, value)
        if self.config.strip_control_characters:
            value = "".join(
                " " if unicodedata.category(character) in {"Cc", "Cf"} else character
                for character in value
            )
        if self.config.lowercase:
            value = value.lower()
        if self.config.normalize_whitespace:
            value = WHITESPACE_PATTERN.sub(" ", value).strip()
        return value


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    word_segmenter: str = "none"
    vncorenlp_jar_path: str | None = None
    segmenter_max_characters: int = 20_000
    stopword_preset: str = "none"


class TokenProcessor(Protocol):
    def __call__(self, text: str) -> list[str]: ...

    def metadata(self) -> dict[str, Any]: ...


class TokenizingTextProcessor:
    """Normalize, optionally segment Vietnamese words, then emit BM25 terms."""

    def __init__(
        self,
        normalizer: TextNormalizer,
        tokenizer: TokenizerConfig,
    ) -> None:
        if tokenizer.word_segmenter not in {"none", "pyvi", "vncorenlp"}:
            raise ValueError(
                "tokenizer.word_segmenter must be 'none', 'pyvi', or 'vncorenlp'"
            )
        if tokenizer.segmenter_max_characters <= 0:
            raise ValueError("tokenizer.segmenter_max_characters must be positive")
        if tokenizer.stopword_preset not in STOPWORD_PRESETS:
            raise ValueError(
                f"Unknown tokenizer.stopword_preset={tokenizer.stopword_preset!r}; "
                f"expected one of {sorted(STOPWORD_PRESETS)}"
            )
        self.normalizer = normalizer
        self.config = tokenizer
        self._segmenter: Any = None
        self._stopwords = set(STOPWORD_PRESETS[tokenizer.stopword_preset])

    def _segment(self, text: str) -> str:
        backend = self.config.word_segmenter
        if backend == "none":
            return text
        if backend == "pyvi":
            try:
                from pyvi import ViTokenizer
            except ImportError as error:
                raise RuntimeError("Install Vietnamese tokenizers with: pip install -e '.[vietnamese]'") from error
            return str(ViTokenizer.tokenize(text))
        if not self.config.vncorenlp_jar_path:
            raise ValueError("tokenizer.vncorenlp_jar_path is required for VnCoreNLP")
        if self._segmenter is None:
            try:
                from vncorenlp import VnCoreNLP
            except ImportError as error:
                raise RuntimeError("Install Vietnamese tokenizers with: pip install -e '.[vietnamese]'") from error
            self._segmenter = VnCoreNLP(
                self.config.vncorenlp_jar_path,
                annotators="wseg",
                max_heap_size="-Xmx2g",
            )
        tokens: list[str] = []
        for chunk in _bounded_text_chunks(text, self.config.segmenter_max_characters):
            tokens.extend(
                token for sentence in self._segmenter.tokenize(chunk) for token in sentence
            )
        return " ".join(tokens)

    def _raw_tokens(self, text: str) -> list[str]:
        return WORD_PATTERN.findall(text)

    def __call__(self, text: str) -> list[str]:
        normalized = self.normalizer(text)
        segmented = self._segment(normalized)
        tokens = self._raw_tokens(segmented)
        return [token for token in tokens if token not in self._stopwords]

    def metadata(self) -> dict[str, Any]:
        tokenizer = asdict(self.config)
        return {"normalization": asdict(self.normalizer.config), "tokenizer": tokenizer}


def build_token_processor(config: dict[str, Any] | None = None) -> TokenizingTextProcessor:
    """Build a processor from YAML-friendly mappings, preserving old BM25 defaults."""
    values = config or {}
    normalization = values.get("normalization", {})
    tokenizer = values.get("tokenizer", {})
    if not isinstance(normalization, dict) or not isinstance(tokenizer, dict):
        raise ValueError("preprocessing normalization and tokenizer must be YAML mappings")
    normalizer = TextNormalizer(
        TextNormalizerConfig(
            unicode_form=normalization.get("unicode_form", "NFKC"),
            lowercase=bool(normalization.get("lowercase", True)),
            normalize_whitespace=bool(normalization.get("normalize_whitespace", True)),
            strip_control_characters=bool(
                normalization.get("strip_control_characters", True)
            ),
        )
    )
    return TokenizingTextProcessor(
        normalizer,
        TokenizerConfig(
            word_segmenter=str(tokenizer.get("word_segmenter", "none")),
            vncorenlp_jar_path=tokenizer.get("vncorenlp_jar_path"),
            segmenter_max_characters=int(
                tokenizer.get("segmenter_max_characters", 20_000)
            ),
            stopword_preset=str(tokenizer.get("stopword_preset", "none")),
        ),
    )


def _bounded_text_chunks(text: str, max_characters: int) -> list[str]:
    """Split oversized segmenter requests at whitespace where possible."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_characters, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end + 1)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks
