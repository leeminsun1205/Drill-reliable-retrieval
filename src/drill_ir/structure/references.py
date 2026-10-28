"""Deterministic extraction of explicit article-level legal references."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence


ARTICLE_PATTERN = re.compile(r"\bĐiều\s+(\d{1,3})\b", re.IGNORECASE)
LAW_ID_PATTERN = re.compile(
    r"\b([0-9]{1,4}/[0-9]{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*)\b", re.IGNORECASE
)
LOCAL_PATTERN = re.compile(
    r"(?:của|tại)?\s*(?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|"
    r"Quyết\s+định|văn\s+bản)?\s+này\b",
    re.IGNORECASE,
)
NAMED_INSTRUMENT_PATTERN = re.compile(
    r"(?:của\s+)?(?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|"
    r"Quyết\s+định)\s+(?!này\b|số\b)[A-ZĐÀ-Ỹ]",
    re.IGNORECASE,
)
AMENDMENT_PATTERN = re.compile(
    r"(?:sửa\s+đổi|bổ\s+sung|bãi\s+bỏ|thay\s+thế)[^.;:\n]{0,60}$",
    re.IGNORECASE,
)
INSTRUMENT_PATTERN = re.compile(
    r"\b(?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)\b",
    re.IGNORECASE,
)
QUOTED_HEADING_PATTERN = re.compile(
    r"như\s+sau\s*[:：]?\s*[“\"]?\s*$",
    re.IGNORECASE,
)


def normalize_law_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).upper()
    normalized = normalized.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", "", normalized)


@dataclass(frozen=True, slots=True)
class ArticleNumberRecord:
    law_id: str
    article_number: int
    aid: int
    status: str
    candidate_eligible: bool


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    source_aid: int
    source_law_id: str
    article_number: int
    target_law_id: str | None
    target_aid: int | None
    scope: str
    status: str
    candidate_eligible: bool
    matched_text: str
    source_char_start: int
    source_char_end: int
    context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_article_number_index(
    documents: Sequence[Mapping[str, Any]],
    *,
    numbering_schemes: Mapping[str, str] | None = None,
    excluded_article_numbers: Mapping[str, Sequence[int]] | None = None,
    missing_body_aids: Sequence[int] = (),
) -> tuple[dict[tuple[str, int], ArticleNumberRecord], list[ArticleNumberRecord]]:
    schemes = {
        normalize_law_id(law_id): scheme
        for law_id, scheme in (numbering_schemes or {}).items()
    }
    excluded = {
        normalize_law_id(law_id): {int(number) for number in numbers}
        for law_id, numbers in (excluded_article_numbers or {}).items()
    }
    missing = {int(aid) for aid in missing_body_aids}
    index: dict[tuple[str, int], ArticleNumberRecord] = {}
    records: list[ArticleNumberRecord] = []
    for document in documents:
        law_id = str(document["law_id"])
        normalized_law_id = normalize_law_id(law_id)
        if schemes.get(normalized_law_id, "articles") != "articles":
            continue
        for position, article in enumerate(document.get("content") or (), start=1):
            if position in excluded.get(normalized_law_id, set()):
                continue
            aid = int(article["aid"])
            is_missing = aid in missing
            record = ArticleNumberRecord(
                law_id=law_id,
                article_number=position,
                aid=aid,
                status="missing_body" if is_missing else "position_verified",
                candidate_eligible=not is_missing,
            )
            index[(normalized_law_id, position)] = record
            records.append(record)
    return index, records


def extract_reference_records(
    documents: Sequence[Mapping[str, Any]],
    article_index: Mapping[tuple[str, int], ArticleNumberRecord],
) -> list[ReferenceRecord]:
    records: list[ReferenceRecord] = []
    for document in documents:
        source_law_id = str(document["law_id"])
        normalized_source = normalize_law_id(source_law_id)
        for article in document.get("content") or ():
            source_aid = int(article["aid"])
            text = str(article.get("content_Article") or "")
            sentence_start = 0
            sentences: list[tuple[str, int]] = []
            for boundary in re.finditer(r"(?<=[.;])\s+|\n+", text):
                sentences.append((text[sentence_start:boundary.start()], sentence_start))
                sentence_start = boundary.end()
            sentences.append((text[sentence_start:], sentence_start))
            for sentence, sentence_offset in sentences:
                mentions = list(ARTICLE_PATTERN.finditer(sentence))
                if not mentions:
                    continue
                for mention in mentions:
                    number = int(mention.group(1))
                    global_start = sentence_offset + mention.start()
                    global_end = sentence_offset + mention.end()
                    context = re.sub(
                        r"\s+", " ", text[max(0, global_start - 180):global_end + 300]
                    ).strip()
                    suffix = sentence[mention.end() : mention.end() + 220]
                    local_match = LOCAL_PATTERN.search(suffix)
                    named_match = NAMED_INSTRUMENT_PATTERN.search(suffix)
                    explicit_after = [
                        match for match in LAW_ID_PATTERN.finditer(sentence)
                        if mention.end() <= match.start() <= mention.end() + 220
                    ]
                    nearest_after = explicit_after[0] if explicit_after else None
                    next_mention = next(
                        (candidate for candidate in mentions if candidate.start() > mention.start()),
                        None,
                    )
                    explicit_after_is_nested = (
                        nearest_after is not None
                        and next_mention is not None
                        and next_mention.start() < nearest_after.start()
                        and "(" in sentence[mention.end():nearest_after.start()]
                    )
                    competing_named_instrument = False
                    if nearest_after is not None and named_match is not None:
                        instruments = list(
                            INSTRUMENT_PATTERN.finditer(
                                sentence[mention.end():nearest_after.start()]
                            )
                        )
                        competing_named_instrument = bool(
                            instruments
                            and instruments[-1].start() > named_match.start() + 8
                        )
                    local_is_direct = local_match is not None and local_match.start() <= 80
                    named_precedes_local = (
                        named_match is not None
                        and (local_match is None or named_match.start() < local_match.start())
                    )
                    amendment_target = bool(
                        AMENDMENT_PATTERN.search(sentence[max(0, mention.start() - 80):mention.start()])
                    )
                    quoted_heading = bool(
                        QUOTED_HEADING_PATTERN.search(
                            sentence[max(0, mention.start() - 40):mention.start()]
                        )
                        or (
                            re.match(r"\s*\.\s*[A-ZĐÀ-Ỹ]", sentence[mention.end():])
                            is not None
                            and (
                                mention.start() == 0
                                or sentence[mention.start() - 1] in '.:“\"”'
                            )
                        )
                    )
                    if quoted_heading:
                        records.append(
                            ReferenceRecord(
                                source_aid, source_law_id, number, None, None,
                                "unresolved", "quoted_article_heading", False,
                                mention.group(0), global_start, global_end, context,
                            )
                        )
                        continue
                    if local_is_direct and not named_precedes_local:
                        target_law_id = normalized_source
                        scope = "same_document"
                    elif (
                        nearest_after is not None
                        and not explicit_after_is_nested
                        and not competing_named_instrument
                    ):
                        target_law_id = normalize_law_id(nearest_after.group(1))
                        scope = "cross_document"
                    elif named_precedes_local or competing_named_instrument:
                        records.append(
                            ReferenceRecord(
                                source_aid, source_law_id, number, None, None,
                                "unresolved", "named_target_without_law_id", False,
                                mention.group(0), global_start, global_end, context,
                            )
                        )
                        continue
                    elif amendment_target:
                        records.append(
                            ReferenceRecord(
                                source_aid, source_law_id, number, None, None,
                                "unresolved", "amendment_target_without_law_id", False,
                                mention.group(0), global_start, global_end, context,
                            )
                        )
                        continue
                    elif len({normalize_law_id(value) for value in LAW_ID_PATTERN.findall(sentence)}) > 1:
                        records.append(
                            ReferenceRecord(
                                source_aid, source_law_id, number, None, None,
                                "ambiguous", "multiple_target_law_ids", False,
                                mention.group(0), global_start, global_end, context,
                            )
                        )
                        continue
                    else:
                        records.append(
                            ReferenceRecord(
                                source_aid, source_law_id, number, None, None,
                                "unresolved", "missing_target_law_id", False,
                                mention.group(0), global_start, global_end, context,
                            )
                        )
                        continue
                    target = article_index.get((target_law_id, number))
                    records.append(
                        ReferenceRecord(
                            source_aid=source_aid,
                            source_law_id=source_law_id,
                            article_number=number,
                            target_law_id=target.law_id if target else target_law_id,
                            target_aid=target.aid if target else None,
                            scope=scope,
                            status=(target.status if target else "target_not_in_corpus"),
                            candidate_eligible=(target.candidate_eligible if target else False),
                            matched_text=mention.group(0),
                            source_char_start=global_start,
                            source_char_end=global_end,
                            context=context,
                        )
                    )
    return records
