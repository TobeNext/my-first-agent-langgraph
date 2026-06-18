from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.domain.resume_parser import (
    ParsedResumeMarkdown,
    parse_resume_markdown,
    parse_resume_sections,
)
from app.schemas.interview_start import InterviewStartRequest

RESUME_MARKDOWN_MARKER = "Resume Markdown:"
JOB_DESCRIPTION_MARKDOWN_MARKER = "Job Description Markdown:"


def extract_structured_interview_start_request(raw: str) -> InterviewStartRequest | None:
    try:
        return InterviewStartRequest.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def detect_kickoff_payload_format(raw: str) -> str:
    if extract_structured_interview_start_request(raw):
        return "structured-start-v1"
    if (
        RESUME_MARKDOWN_MARKER in raw
        or JOB_DESCRIPTION_MARKDOWN_MARKER in raw
        or re.search(r"Selected interview direction:", raw, flags=re.IGNORECASE)
    ):
        return "legacy-kickoff"
    return "freeform"


def extract_resume_markdown_from_kickoff_message(raw: str) -> str:
    structured = extract_structured_interview_start_request(raw)
    if structured:
        return structured.resumeMarkdown
    marker_index = raw.find(RESUME_MARKDOWN_MARKER)
    if marker_index < 0:
        return ""
    end_index = raw.find(JOB_DESCRIPTION_MARKDOWN_MARKER, marker_index)
    if end_index < 0:
        end_index = len(raw)
    return raw[marker_index + len(RESUME_MARKDOWN_MARKER) : end_index].strip()


def extract_job_description_markdown_from_kickoff_message(raw: str) -> str:
    structured = extract_structured_interview_start_request(raw)
    if structured:
        return structured.jobDescriptionMarkdown
    marker_index = raw.find(JOB_DESCRIPTION_MARKDOWN_MARKER)
    if marker_index < 0:
        return ""
    return raw[marker_index + len(JOB_DESCRIPTION_MARKDOWN_MARKER) :].strip()


def extract_parsed_resume_from_kickoff_message(raw: str) -> ParsedResumeMarkdown:
    structured = extract_structured_interview_start_request(raw)
    if structured and structured.resumeSections:
        return parse_resume_sections(structured.resumeSections.model_dump())
    return parse_resume_markdown(extract_resume_markdown_from_kickoff_message(raw))


def extract_selected_direction_from_kickoff_message(raw: str) -> str:
    match = re.search(r"Selected interview direction:\s*(.+)", raw, flags=re.IGNORECASE)
    selected = match.group(1).strip() if match else ""
    if selected and selected.lower() != "unknown":
        return selected
    return "通用技术岗位" if re.search(r"[\u3400-\u9fff]", raw) else "General Technical Role"
