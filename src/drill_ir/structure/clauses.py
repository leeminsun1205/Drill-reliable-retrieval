"""Conservative deterministic extraction of Vietnamese numbered clauses (Khoản)."""

from __future__ import annotations

from dataclasses import dataclass
import re


# A marker must begin the article or a physical line, use a small positive
# integer, and be followed by horizontal whitespace. This deliberately rejects
# decimals (4.1), years embedded in prose, and glued forms such as ``2.Phát``.
CLAUSE_MARKER = re.compile(
    r"(?:\A|(?<=\n))[ \t]*(?P<number>[1-9]\d{0,2})[.)](?=[ \t])"
)


@dataclass(frozen=True, slots=True)
class ClauseSpan:
    number: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class ClauseParseResult:
    status: str
    spans: tuple[ClauseSpan, ...]
    detected_numbers: tuple[int, ...]


def parse_clause_spans(text: str) -> ClauseParseResult:
    """Return contiguous clause spans or a conservative fallback status."""
    matches = list(CLAUSE_MARKER.finditer(text))
    numbers = tuple(int(match.group("number")) for match in matches)
    if len(matches) < 2:
        return ClauseParseResult("no_structure", (), numbers)
    if any(current != previous + 1 for previous, current in zip(numbers, numbers[1:])):
        return ClauseParseResult("ambiguous_sequence", (), numbers)

    spans = []
    for index, match in enumerate(matches):
        start = 0 if index == 0 else match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append(ClauseSpan(number=numbers[index], char_start=start, char_end=end))
    return ClauseParseResult("parsed", tuple(spans), numbers)
