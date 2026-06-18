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

_LEGACY_METADATA_KEYS = {"mainCategory", "subCategory", "company", "tags"}
_SCALAR_METADATA_KEYS = {"role", "difficulty", "skillArea"}


def normalize_skill_area_from_text(text: str | None) -> list[str]:
    if not text:
        return [DEFAULT_SKILL_AREA]
    matched = [skill for pattern, skill in _SKILL_RULES if pattern.search(text)]
    return _unique(matched) or [DEFAULT_SKILL_AREA]


def normalize_skill_area_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return [DEFAULT_SKILL_AREA]
    explicit = _split_skill_area(metadata.get("skillArea"))
    if explicit:
        return explicit
    text = " ".join(
        str(metadata.get(key) or "")
        for key in ("question", "answer", "text", "mainCategory", "subCategory", "tags")
    )
    return normalize_skill_area_from_text(text)


def clean_interview_question_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    skill_area = normalize_skill_area_from_metadata(metadata)
    cleaned_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in _LEGACY_METADATA_KEYS and key not in _SCALAR_METADATA_KEYS
    }
    return {
        "question": str(metadata.get("question") or metadata.get("text") or "").strip(),
        "answer": metadata.get("answer"),
        "questionType": metadata.get("questionType") or "knowledge-check",
        "source": metadata.get("source") or "interview-question-bank",
        "sourceFile": metadata.get("sourceFile"),
        "text": metadata.get("text"),
        "metadata": cleaned_metadata,
        "role": metadata.get("role") or "general",
        "difficulty": metadata.get("difficulty") or "medium",
        "skillArea": skill_area,
    }


def build_skill_area_audit(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(normalize_skill_area_from_metadata(record))
    return dict(sorted(counter.items()))


def format_skill_area(value: Any) -> list[str]:
    return _split_skill_area(value)


def _split_skill_area(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return _unique(item for item in re.split(r"[\s,]+", value) if item)
    return []


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
