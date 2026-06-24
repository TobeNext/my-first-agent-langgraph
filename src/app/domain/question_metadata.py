from __future__ import annotations

import re
from collections import Counter
from typing import Any

DEFAULT_SKILL_AREA = "agent"

_SKILL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Java|后端|JVM", re.IGNORECASE), "java"),
    (re.compile(r"Spring(?: Boot| Cloud)?", re.IGNORECASE), "spring"),
    (re.compile(r"TypeScript|\bTS\b|Node(?:\.js)?|NestJS", re.IGNORECASE), "typescript"),
    (re.compile(r"Vue|前端", re.IGNORECASE), "vue"),
    (re.compile(r"Mastra", re.IGNORECASE), "mastra"),
    (re.compile(r"LangChain", re.IGNORECASE), "langchain"),
    (re.compile(r"CrewAI", re.IGNORECASE), "crewai"),
    (re.compile(r"RAG|检索|召回|向量", re.IGNORECASE), "rag"),
    (re.compile(r"Milvus|向量数据库|vector database", re.IGNORECASE), "milvus"),
    (re.compile(r"BM25|rerank|重排", re.IGNORECASE), "bm25"),
    (re.compile(r"Memory|记忆|上下文", re.IGNORECASE), "memory"),
    (re.compile(r"Tool|Function Call|MCP|工具调用", re.IGNORECASE), "tool-calling"),
    (re.compile(r"Multi-Agent|多\s*Agent|多智能体", re.IGNORECASE), "multi-agent"),
    (re.compile(r"Workflow|工作流", re.IGNORECASE), "workflow"),
    (re.compile(r"路由|fallback|成本|小模型|大模型", re.IGNORECASE), "model-routing"),
    (re.compile(r"Docker|Kubernetes|K8s", re.IGNORECASE), "docker"),
    (re.compile(r"微服务|API Gateway|网关", re.IGNORECASE), "microservices"),
    (re.compile(r"观测|日志|trace|监控", re.IGNORECASE), "observability"),
)

_LEGACY_METADATA_KEYS = {"mainCategory", "subCategory", "company"}
_SCALAR_METADATA_KEYS = {
    "role",
    "difficulty",
    "skillArea",
    "skills",
    "level",
    "questionType",
    "question_type",
    "answer_points",
    "answerPoints",
    "job_family",
    "jobFamily",
    "job_duties",
    "jobDuties",
    "language",
    "embedding_text",
    "embeddingText",
    "source",
    "sourceFile",
    "text",
    "question",
    "answer",
    "tags",
    "isActive",
    "userId",
}


def normalize_skill_area_from_text(text: str | None) -> list[str]:
    if not text:
        return [DEFAULT_SKILL_AREA]
    matched = [skill for pattern, skill in _SKILL_RULES if pattern.search(text)]
    return _unique(matched) or [DEFAULT_SKILL_AREA]


def normalize_skill_area_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return [DEFAULT_SKILL_AREA]
    explicit = _split_skill_area(metadata.get("skills") or metadata.get("skillArea"))
    if explicit:
        return explicit
    text = " ".join(
        str(metadata.get(key) or "")
        for key in (
            "question",
            "answer",
            "text",
            "mainCategory",
            "subCategory",
            "tags",
            "job_duties",
            "jobDuties",
        )
    )
    return normalize_skill_area_from_text(text)


def clean_interview_question_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    skill_area = normalize_skill_area_from_metadata(metadata)
    question_type = _normalize_question_type(
        metadata.get("question_type") or metadata.get("questionType")
    )
    level = _normalize_level(metadata.get("level") or metadata.get("difficulty"))
    difficulty = _difficulty_score(metadata.get("difficulty") or metadata.get("level"))
    answer_points = _answer_points(
        metadata.get("answer_points") or metadata.get("answerPoints"),
        metadata.get("answer"),
    )
    tags = _string_list(metadata.get("tags"))
    job_duties = _string_list(metadata.get("job_duties") or metadata.get("jobDuties"))
    embedding_text = str(
        metadata.get("embedding_text") or metadata.get("embeddingText") or ""
    ).strip()
    cleaned_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in _LEGACY_METADATA_KEYS and key not in _SCALAR_METADATA_KEYS
    }
    return {
        "question": str(metadata.get("question") or metadata.get("text") or "").strip(),
        "answer": metadata.get("answer"),
        "answer_points": answer_points,
        "tags": tags,
        "skills": skill_area,
        "level": level,
        "question_type": question_type,
        "job_family": str(metadata.get("job_family") or metadata.get("jobFamily") or "").strip(),
        "job_duties": job_duties,
        "language": str(metadata.get("language") or "zh").strip() or "zh",
        "embedding_text": embedding_text,
        "questionType": question_type,
        "source": metadata.get("source") or "interview-question-bank",
        "sourceFile": metadata.get("sourceFile"),
        "text": metadata.get("text"),
        "metadata": cleaned_metadata,
        "role": metadata.get("role") or "general",
        "difficulty": difficulty,
        "skillArea": skill_area,
        "isActive": _bool_or_default(metadata.get("isActive"), default=True),
        "userId": str(metadata.get("userId") or "global").strip() or "global",
    }


def build_skill_area_audit(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(normalize_skill_area_from_metadata(record))
    return dict(sorted(counter.items()))


def format_skill_area(value: Any) -> list[str]:
    return _split_skill_area(value)


def normalize_question_type(value: Any) -> str:
    return _normalize_question_type(value)


def normalize_level(value: Any) -> str:
    return _normalize_level(value)


def normalize_difficulty_score(value: Any) -> int:
    return _difficulty_score(value)


def _split_skill_area(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return _unique(item for item in re.split(r"[\s,]+", value) if item)
    return []


def _normalize_question_type(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "scenario": "case_analysis",
        "knowledge-check": "knowledge_check",
        "knowledge_check": "knowledge_check",
        "system_design": "system_design",
        "system-design": "system_design",
        "technical": "technical",
        "experience_probe": "experience_probe",
        "experience-probe": "experience_probe",
        "case_analysis": "case_analysis",
        "case-analysis": "case_analysis",
        "culture_fit": "culture_fit",
        "culture-fit": "culture_fit",
    }
    return aliases.get(normalized, "knowledge_check")


def _normalize_level(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"junior", "entry", "easy"}:
        return "junior"
    if normalized in {"senior", "hard", "expert"}:
        return "senior"
    if normalized in {"middle", "mid", "medium", "intermediate"}:
        return "middle"
    return "unknown"


def _legacy_difficulty(value: str) -> str:
    if value == "senior":
        return "hard"
    if value == "junior":
        return "easy"
    return "medium" if value in {"middle", "unknown"} else value


def _difficulty_score(value: Any) -> int:
    if isinstance(value, int):
        return min(10, max(1, value))
    if isinstance(value, float):
        return min(10, max(1, round(value)))
    normalized = str(value or "").strip().lower()
    if normalized in {"easy", "junior", "entry"}:
        return 3
    if normalized in {"medium", "middle", "mid", "intermediate"}:
        return 6
    if normalized in {"hard", "senior", "expert"}:
        return 8
    return 5


def _bool_or_default(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _answer_points(value: Any, answer: Any) -> list[str]:
    explicit = _string_list(value)
    if explicit:
        return explicit
    if not isinstance(answer, str):
        return []
    return [
        re.sub(r"^(?:[-*+•]\s+|\d+[.)]\s+)", "", line).strip()
        for line in answer.splitlines()
        if line.strip()
    ][:8]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        try:
            parsed = value.split(",")
        except AttributeError:
            parsed = [value]
        return _unique(item.strip() for item in parsed if item.strip())
    return []


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
