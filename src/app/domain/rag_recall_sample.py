from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.domain.question_retriever import RagRecallTrace
from app.schemas.interview_state import InterviewSessionState, InterviewTopicNodeState, RoundType

RAG_LOG_SCHEMA_VERSION = 1


def write_initialization_rag_recall_sample(
    *,
    thread_id: str,
    target_role: str,
    recall_traces: list[RagRecallTrace],
    state: InterviewSessionState,
    rag_log_root: str | Path | None = None,
) -> str:
    created_at = _now()
    root = _rag_log_root(rag_log_root)
    file_name = f"{_sanitize_timestamp(created_at)}-{thread_id}-rag-recall-sample.json"
    file_path = root / file_name
    sample = _build_sample(
        thread_id=thread_id,
        target_role=target_role,
        created_at=created_at,
        updated_at=created_at,
        recall_traces=recall_traces,
        state=state,
    )
    _write_json(file_path, sample)
    return str(file_path)


def update_rag_recall_sample_answer_performance(
    file_path: str | Path,
    state: InterviewSessionState,
) -> None:
    path = Path(file_path)
    current = json.loads(path.read_text(encoding="utf-8"))
    answer_performance_map = _answer_performance_map(state)
    recalls = []
    for trace in current.get("recalls", []):
        performances = [
            answer_performance_map[_normalize_question_text(selection["questionText"])]
            for selection in trace.get("finalSelectedQuestions", [])
            if _normalize_question_text(selection.get("questionText", "")) in answer_performance_map
        ]
        recalls.append({**trace, "postInterviewAnswerPerformance": performances})
    current.update(
        {
            "updatedAt": _now(),
            "recalls": recalls,
            "interviewSnapshot": {
                "phase": state.phase,
                "finalReportReady": state.finalReportReady,
                "answerPerformances": _answer_performance_list(state),
            },
        }
    )
    _write_json(path, current)


def _build_sample(
    *,
    thread_id: str,
    target_role: str,
    created_at: str,
    updated_at: str,
    recall_traces: list[RagRecallTrace],
    state: InterviewSessionState,
) -> dict[str, Any]:
    answer_performance_map = _answer_performance_map(state)
    recalls = []
    for trace in recall_traces:
        trace_payload = _serialize(trace)
        performances = [
            answer_performance_map[_normalize_question_text(selection["questionText"])]
            for selection in trace_payload["finalSelectedQuestions"]
            if _normalize_question_text(selection["questionText"]) in answer_performance_map
        ]
        recalls.append({**trace_payload, "postInterviewAnswerPerformance": performances})
    return {
        "schemaVersion": RAG_LOG_SCHEMA_VERSION,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "threadId": thread_id,
        "targetRole": target_role,
        "recalls": recalls,
        "interviewSnapshot": {
            "phase": state.phase,
            "finalReportReady": state.finalReportReady,
            "answerPerformances": _answer_performance_list(state),
        },
    }


def _answer_performance_list(state: InterviewSessionState) -> list[dict[str, Any]]:
    return [
        _question_answer_performance(round_item.type, node)
        for round_item in state.rounds
        for node in round_item.nodes
    ]


def _answer_performance_map(state: InterviewSessionState) -> dict[str, dict[str, Any]]:
    return {
        _normalize_question_text(performance["mainQuestion"]): performance
        for performance in _answer_performance_list(state)
    }


def _question_answer_performance(
    round_type: RoundType,
    node: InterviewTopicNodeState,
) -> dict[str, Any]:
    latest_attempt = node.answerAttempts[-1] if node.answerAttempts else None
    return {
        "roundType": round_type,
        "topic": node.topic,
        "mainQuestion": node.mainQuestion,
        "nodeStatus": node.status,
        "followUpCount": node.followUpCount,
        "answerAttemptCount": len(node.answerAttempts),
        "latestClassification": latest_attempt.classification if latest_attempt else None,
        "aggregatedScore": node.aggregatedScore,
        "strengths": node.summary.strengths if node.summary else [],
        "missingPoints": node.summary.missingPoints if node.summary else [],
        "incorrectPoints": node.summary.weaknesses if node.summary else [],
    }


def _rag_log_root(root: str | Path | None) -> Path:
    path = Path(root) if root is not None else Path(get_settings().rag_log_root)
    return path.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _normalize_question_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sanitize_timestamp(timestamp: str) -> str:
    return timestamp.replace(":", "-").replace(".", "-")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
