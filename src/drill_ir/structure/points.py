"""Conservative deterministic extraction of Vietnamese lettered points (Điểm)."""

from __future__ import annotations

from dataclasses import dataclass
import re


# Vietnamese legal lists conventionally omit f, j, w, and z and place đ after d.
# A marker must appear at the span start or after a physical/list separator. The
# separator itself is deliberately excluded from the point span.
POINT_LETTERS = ("a", "b", "c", "d", "đ", "e", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y")
POINT_MARKER = re.compile(
    r"(?:\A|(?<=[\n\t:;]))[ ]*(?P<marker>(?P<letter>[abcdđeghiklmnopqrstuvxy])\))(?=[ \t])",
    flags=re.IGNORECASE,
)
_POINT_INDEX = {letter: index for index, letter in enumerate(POINT_LETTERS)}


@dataclass(frozen=True, slots=True)
class PointSpan:
    letter: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class PointParseResult:
    status: str
    lead_in_end: int
    spans: tuple[PointSpan, ...]
    detected_letters: tuple[str, ...]


def parse_point_spans(text: str) -> PointParseResult:
    """Return point spans inside one accepted Khoản span."""
    matches = list(POINT_MARKER.finditer(text))
    letters = tuple(match.group("letter").lower() for match in matches)
    if len(matches) < 2:
        return PointParseResult("no_structure", 0, (), letters)
    indices = tuple(_POINT_INDEX[letter] for letter in letters)
    if any(current != previous + 1 for previous, current in zip(indices, indices[1:])):
        return PointParseResult("ambiguous_sequence", 0, (), letters)

    starts = [match.start("marker") for match in matches]
    spans = tuple(
        PointSpan(
            letter=letters[index],
            char_start=start,
            char_end=starts[index + 1] if index + 1 < len(starts) else len(text),
        )
        for index, start in enumerate(starts)
    )
    return PointParseResult("parsed", starts[0], spans, letters)
