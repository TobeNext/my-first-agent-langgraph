from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.integrations.models import (
    create_chat_model,
    invoke_json_output_model,
    should_use_native_structured_output,
)

MatchPriority = Literal["low", "medium", "high"]
MatchType = Literal["skill", "responsibility", "domain", "project-evidence"]
ResumeOnlyCategory = Literal["skill", "project", "domain"]
JdOnlyCategory = Literal["responsibility", "requirement", "preferred", "domain"]
SuggestedQuestionType = Literal[
    "system_design",
    "experience_probe",
    "case_analysis",
    "knowledge_check",
]
GENERIC_MATCH_TOKENS = {
    "开发",
    "项目",
    "经验",
    "负责",
    "熟悉",
    "具备",
    "能力",
    "岗位",
    "要求",
}


class ResumeJdMatchEvidence(BaseModel):
    resumeSignals: list[str] = Field(default_factory=list)
    jobSignals: list[str] = Field(default_factory=list)
    projectSignals: list[str] = Field(default_factory=list)


class ResumeJdMatchItem(BaseModel):
    resumeSignal: str
    jobSignal: str
    matchType: MatchType
    relevance: float = Field(ge=0, le=1)
    priority: MatchPriority
    evidence: ResumeJdMatchEvidence = Field(default_factory=ResumeJdMatchEvidence)
    interviewFocus: list[str] = Field(default_factory=list)
    suggestedQuestionTypes: list[SuggestedQuestionType] = Field(default_factory=list)


class ResumeOnlyItem(BaseModel):
    resumeSignal: str
    category: ResumeOnlyCategory
    evidence: list[str] = Field(default_factory=list)


class JdOnlyItem(BaseModel):
    jobSignal: str
    category: JdOnlyCategory
    priority: MatchPriority
    evidence: list[str] = Field(default_factory=list)


class ResumeJdMatchAnalysis(BaseModel):
    resumeJdMatch: list[ResumeJdMatchItem] = Field(default_factory=list)
    resumeOnly: list[ResumeOnlyItem] = Field(default_factory=list)
    jdOnly: list[JdOnlyItem] = Field(default_factory=list)
    isJobMatched: bool = True
    mismatchReason: str | None = None


def build_resume_jd_match_analysis(
    *,
    professional_skills: str,
    project_experience: str,
    job_description: str,
    normalized_skills: list[str],
    normalized_project_topics: list[str],
    chat_model: Any | None = None,
) -> ResumeJdMatchAnalysis:
    if not job_description.strip():
        return _mock_llm_analysis(
            professional_skills=professional_skills,
            project_experience=project_experience,
            job_description=job_description,
            normalized_skills=normalized_skills,
            normalized_project_topics=normalized_project_topics,
        )

    model = chat_model or create_chat_model()
    prompt = _build_prompt(
        professional_skills=professional_skills,
        project_experience=project_experience,
        job_description=job_description,
        normalized_skills=normalized_skills,
        normalized_project_topics=normalized_project_topics,
    )
    try:
        if should_use_native_structured_output(model):
            structured_model = model.with_structured_output(ResumeJdMatchAnalysis)
            result = structured_model.invoke(prompt)
            return _normalize_analysis(result)
        raw = invoke_json_output_model(model, prompt)
        return _normalize_analysis(_parse_raw_model_json(raw))
    except Exception:
        return _mock_llm_analysis(
            professional_skills=professional_skills,
            project_experience=project_experience,
            job_description=job_description,
            normalized_skills=normalized_skills,
            normalized_project_topics=normalized_project_topics,
        )


def _build_prompt(
    *,
    professional_skills: str,
    project_experience: str,
    job_description: str,
    normalized_skills: list[str],
    normalized_project_topics: list[str],
) -> str:
    payload = {
        "resume": {
            "professionalSkills": _compact_text(professional_skills),
            "projectExperience": _compact_text(project_experience),
            "normalizedSkills": normalized_skills[:12],
            "normalizedProjectTopics": normalized_project_topics[:8],
        },
        "jobDescription": _compact_text(job_description, limit=2400),
    }
    return "\n".join(
        [
            "You are matching a candidate resume against a job description.",
            "Return a JSON object that strictly follows this shape:",
            json.dumps(ResumeJdMatchAnalysis.model_json_schema(), ensure_ascii=False),
            "Rules:",
            "- Split the result into exactly three top-level sections.",
            "- resumeJdMatch contains only direct resume/JD matches.",
            "- resumeOnly contains resume evidence not required by the JD.",
            "- jdOnly contains JD requirements not evidenced by the resume.",
            "- If resumeJdMatch is empty and the JD has content, set isJobMatched=false.",
            "- Do not invent evidence that is not present in the input.",
            "Input:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def _normalize_analysis(value: Any) -> ResumeJdMatchAnalysis:
    analysis = (
        value
        if isinstance(value, ResumeJdMatchAnalysis)
        else ResumeJdMatchAnalysis.model_validate(value)
    )
    if analysis.resumeJdMatch:
        return analysis.model_copy(update={"isJobMatched": True, "mismatchReason": None})
    if analysis.jdOnly:
        return analysis.model_copy(
            update={
                "isJobMatched": False,
                "mismatchReason": analysis.mismatchReason
                or "岗位不匹配：简历中没有发现与 JD 直接匹配的技能、职责或项目证据。",
            }
        )
    return analysis.model_copy(update={"isJobMatched": True})


def _mock_llm_analysis(
    *,
    professional_skills: str,
    project_experience: str,
    job_description: str,
    normalized_skills: list[str],
    normalized_project_topics: list[str],
) -> ResumeJdMatchAnalysis:
    resume_signals = _unique([*normalized_skills, *_extract_lines(professional_skills)])
    project_signals = _unique([*normalized_project_topics, *_extract_lines(project_experience)])
    jd_signals = _extract_lines(job_description)
    if not job_description.strip():
        return ResumeJdMatchAnalysis(
            resumeOnly=[
                ResumeOnlyItem(
                    resumeSignal=signal,
                    category="project" if signal in project_signals else "skill",
                    evidence=[signal],
                )
                for signal in _unique([*resume_signals, *project_signals])[:8]
            ],
            isJobMatched=True,
        )

    matches: list[ResumeJdMatchItem] = []
    matched_resume_keys: set[str] = set()
    matched_jd_keys: set[str] = set()
    for resume_signal in resume_signals:
        for jd_signal in jd_signals:
            score = _overlap_score(resume_signal, jd_signal)
            if score <= 0:
                continue
            matches.append(
                ResumeJdMatchItem(
                    resumeSignal=resume_signal,
                    jobSignal=jd_signal,
                    matchType="skill",
                    relevance=score,
                    priority="high" if score >= 0.5 else "medium",
                    evidence=ResumeJdMatchEvidence(
                        resumeSignals=[resume_signal],
                        jobSignals=[jd_signal],
                        projectSignals=_related_project_signals(project_signals, resume_signal),
                    ),
                    interviewFocus=[resume_signal, jd_signal],
                    suggestedQuestionTypes=["experience_probe", "knowledge_check"],
                )
            )
            matched_resume_keys.add(_normalize(resume_signal))
            matched_jd_keys.add(_normalize(jd_signal))
            break

    resume_only = [
        ResumeOnlyItem(
            resumeSignal=signal,
            category="project" if signal in project_signals else "skill",
            evidence=[signal],
        )
        for signal in _unique([*resume_signals, *project_signals])
        if _normalize(signal) not in matched_resume_keys
    ][:8]
    jd_only = [
        JdOnlyItem(
            jobSignal=signal,
            category="requirement",
            priority="medium",
            evidence=[signal],
        )
        for signal in jd_signals
        if _normalize(signal) not in matched_jd_keys
    ][:8]
    return _normalize_analysis(
        ResumeJdMatchAnalysis(
            resumeJdMatch=matches[:8],
            resumeOnly=resume_only,
            jdOnly=jd_only,
            isJobMatched=bool(matches),
            mismatchReason=None
            if matches
            else "岗位不匹配：简历中没有发现与 JD 直接匹配的技能、职责或项目证据。",
        )
    )


def _parse_raw_model_json(value: Any) -> dict[str, Any]:
    content = getattr(value, "content", value)
    text = str(content)
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    json_text = fenced_match.group(1).strip() if fenced_match else _extract_json_object_text(text)
    if not json_text:
        raise ValueError("Model response did not contain a JSON object.")
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def _extract_json_object_text(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def _compact_text(value: str, *, limit: int = 1600) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _extract_lines(value: str) -> list[str]:
    return _unique(
        re.sub(r"^(?:#{1,6}\s*|[-*+•]\s+|\d+[.)]\s+)", "", line).strip()
        for line in value.splitlines()
        if line.strip()
    )


def _overlap_score(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return min(1.0, len(overlap) / max(1, min(len(left_tokens), len(right_tokens))))


def _tokens(value: str) -> list[str]:
    normalized = _normalize(value)
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9_.+#-]+|[\u3400-\u9fff]+", normalized):
        token = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            tokens.extend(_chinese_bigrams(token))
        elif len(token) >= 2:
            tokens.append(token)
    return [token for token in tokens if token not in GENERIC_MATCH_TOKENS]


def _chinese_bigrams(value: str) -> list[str]:
    if len(value) <= 2:
        return [value]
    return [value[index : index + 2] for index in range(len(value) - 1)]


def _related_project_signals(project_signals: list[str], resume_signal: str) -> list[str]:
    resume_tokens = set(_tokens(resume_signal))
    return [
        signal
        for signal in project_signals
        if resume_tokens and resume_tokens & set(_tokens(signal))
    ][:3]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
