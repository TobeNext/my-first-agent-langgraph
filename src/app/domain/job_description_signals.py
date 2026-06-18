from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.domain.resume_parser import extract_normalized_resume_topics

QuestionDriver = Literal["resume", "job-description", "resume-and-job-description"]


@dataclass(frozen=True)
class JobDescriptionSignalSet:
    responsibilities: list[str]
    technicalRequirements: list[str]
    preferredSkills: list[str]
    domainTerms: list[str]
    topSignals: list[str]
    alignedSignals: list[str]
    gapSignals: list[str]
    priorityKeywords: list[str]


ENGLISH_STOP_WORDS = {
    "and",
    "the",
    "with",
    "for",
    "from",
    "into",
    "will",
    "your",
    "have",
    "that",
    "this",
    "using",
    "build",
    "work",
    "team",
}


def resolve_question_driver(
    *,
    has_resume_signals: bool,
    has_job_description_signals: bool,
) -> QuestionDriver:
    if has_resume_signals and has_job_description_signals:
        return "resume-and-job-description"
    if has_job_description_signals:
        return "job-description"
    return "resume"


def extract_job_description_signal_set(
    *,
    job_description: str | None,
    resume_topics: list[str] | None = None,
    project_topics: list[str] | None = None,
) -> JobDescriptionSignalSet:
    responsibilities, technical, preferred, uncategorized = _collect_signal_buckets(
        job_description or ""
    )
    responsibilities = _dedupe(
        [
            *responsibilities,
            *[line for line in uncategorized if not _is_technical_heading(line)],
        ]
    )[:4]
    technical = _dedupe(technical)[:4]
    preferred = _dedupe(preferred)[:3]
    domain_terms = extract_normalized_resume_topics(
        "\n".join(_dedupe([*responsibilities, *technical, *preferred, *uncategorized]))
    )[:4]
    top_signals = _dedupe([*technical, *responsibilities, *preferred, *domain_terms])[:6]
    context_topics = _dedupe([*(resume_topics or []), *(project_topics or [])])
    aligned = [signal for signal in top_signals if _overlaps_with_context(signal, context_topics)]
    gaps = [signal for signal in top_signals if not _overlaps_with_context(signal, context_topics)]
    keywords = _dedupe(
        [keyword for signal in top_signals for keyword in _extract_keywords(signal)]
    )[:8]
    return JobDescriptionSignalSet(
        responsibilities=responsibilities,
        technicalRequirements=technical,
        preferredSkills=preferred,
        domainTerms=domain_terms,
        topSignals=top_signals,
        alignedSignals=aligned,
        gapSignals=gaps,
        priorityKeywords=keywords,
    )


def _collect_signal_buckets(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    responsibilities: list[str] = []
    technical: list[str] = []
    preferred: list[str] = []
    uncategorized: list[str] = []
    active: str | None = None

    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        bucket = _detect_bucket(line)
        if raw_line.lstrip().startswith("#") and bucket:
            active = bucket
            continue
        if raw_line.lstrip().startswith("#"):
            active = None
            continue
        if active == "responsibilities":
            responsibilities.append(line)
        elif active == "technical":
            technical.append(line)
        elif active == "preferred":
            preferred.append(line)
        else:
            uncategorized.append(line)
    return _dedupe(responsibilities), _dedupe(technical), _dedupe(preferred), _dedupe(uncategorized)


def _detect_bucket(line: str) -> str | None:
    if re.search(r"岗位职责|工作职责|职责|responsibilit|what you|you will|job duties", line, re.I):
        return "responsibilities"
    if _is_technical_heading(line):
        return "technical"
    if re.search(r"加分|优先|preferred|nice to have|bonus|plus", line, re.I):
        return "preferred"
    return None


def _is_technical_heading(line: str) -> bool:
    return bool(
        re.search(
            r"任职要求|岗位要求|要求|资格|must have|requirement|qualification|技能要求", line, re.I
        )
    )


def _normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^(?:#{1,6}\s*|[-*+•]\s+|\d+[.)]\s+)", "", value)).strip()


def _extract_keywords(signal: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", signal).strip().lower()
    tokens = [
        token.strip()
        for token in re.split(r"[^a-z0-9\u3400-\u9fff+#.-]+", normalized)
        if len(token.strip()) >= 4 or re.search(r"[\u3400-\u9fff]", token)
    ]
    return _dedupe([normalized, *[token for token in tokens if token not in ENGLISH_STOP_WORDS]])


def _overlaps_with_context(signal: str, topics: list[str]) -> bool:
    signal_keywords = _extract_keywords(signal)
    for topic in [item.lower() for item in topics if item.strip()]:
        if any(keyword in topic or topic in keyword for keyword in signal_keywords):
            return True
        if any(keyword in signal_keywords for keyword in _extract_keywords(topic)):
            return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
