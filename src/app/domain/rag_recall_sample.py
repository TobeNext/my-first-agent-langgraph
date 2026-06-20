from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.domain.question_critic import QuestionJudgeRecord
from app.domain.question_generator import GeneratedQuestionRecord
from app.domain.question_retriever import RagRecallTrace
from app.schemas.interview_state import InterviewSessionState, InterviewTopicNodeState, RoundType

RAG_LOG_SCHEMA_VERSION = 1


def write_initialization_rag_recall_sample(
    *,
    thread_id: str,
    target_role: str,
    recall_traces: list[RagRecallTrace],
    state: InterviewSessionState,
    generation_trace: list[GeneratedQuestionRecord] | None = None,
    judge_trace: list[QuestionJudgeRecord] | None = None,
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
        generation_trace=generation_trace or [],
        judge_trace=judge_trace or [],
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
    generation_trace: list[GeneratedQuestionRecord],
    judge_trace: list[QuestionJudgeRecord],
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
        "questionSelectionDebug": _question_selection_debug(
            state=state,
            recall_traces=recall_traces,
            generation_trace=generation_trace,
            judge_trace=judge_trace,
        ),
        "interviewSnapshot": {
            "phase": state.phase,
            "finalReportReady": state.finalReportReady,
            "answerPerformances": _answer_performance_list(state),
        },
    }


def _question_selection_debug(
    *,
    state: InterviewSessionState,
    recall_traces: list[RagRecallTrace],
    generation_trace: list[GeneratedQuestionRecord],
    judge_trace: list[QuestionJudgeRecord],
) -> list[dict[str, Any]]:
    generation_payload = [_serialize(record) for record in generation_trace]
    judge_payload = [_serialize(record) for record in judge_trace]
    generation_by_question = {
        _normalize_question_text(record.get("questionText", "")): record
        for record in generation_payload
        if record.get("questionText")
    }
    judge_by_final_question = {
        _normalize_question_text(record.get("finalQuestionText", "")): record
        for record in judge_payload
        if record.get("finalQuestionText")
    }
    recalled_selected_questions = {
        _normalize_question_text(selection.get("questionText", ""))
        for trace in recall_traces
        for selection in _serialize(trace).get("finalSelectedQuestions", [])
        if selection.get("questionText")
    }

    records: list[dict[str, Any]] = []
    for round_item in state.rounds:
        for node in round_item.nodes:
            normalized_question = _normalize_question_text(node.mainQuestion)
            generation = generation_by_question.get(normalized_question, {})
            judge = judge_by_final_question.get(normalized_question, {})
            generation_source = generation.get("source")
            judge_verdict = judge.get("verdict")
            retrieved_question_matched = normalized_question in recalled_selected_questions
            selection_source = _selection_source(
                generation_source=generation_source,
                judge_verdict=judge_verdict,
                retrieved_question_matched=retrieved_question_matched,
            )
            records.append(
                {
                    "roundType": round_item.type,
                    "nodeId": node.id,
                    "topic": node.topic,
                    "mainQuestion": node.mainQuestion,
                    "selectionSource": selection_source,
                    "fallbackReason": _fallback_reason(
                        selection_source=selection_source,
                        judge_verdict=judge_verdict,
                    ),
                    "failureReasons": judge.get("failureReasons", []),
                    "generationSource": generation_source,
                    "judgeVerdict": judge_verdict,
                    "retrievedQuestionMatched": retrieved_question_matched,
                    "questionId": generation.get("questionId") or judge.get("questionId"),
                    "originalQuestionText": judge.get("originalQuestionText"),
                    "finalQuestionText": judge.get("finalQuestionText") or node.mainQuestion,
                    "questionDriver": generation.get("questionDriver"),
                    "targetAbility": generation.get("targetAbility"),
                    "resumeSignals": generation.get("resumeSignals", []),
                    "jobDescriptionSignals": generation.get("jobDescriptionSignals", []),
                    "selectionReason": generation.get("selectionReason"),
                }
            )
    return records


def _selection_source(
    *,
    generation_source: Any,
    judge_verdict: Any,
    retrieved_question_matched: bool,
) -> str:
    if judge_verdict == "fallback" or generation_source == "fallback":
        return "fallback"
    if generation_source == "retrieved" or retrieved_question_matched:
        return "retrieved"
    return "unknown"


def _fallback_reason(*, selection_source: str, judge_verdict: Any) -> str | None:
    if selection_source != "fallback":
        return None
    if judge_verdict == "fallback":
        return "judge-replaced-candidate"
    return "retrieval-empty-filled-by-fallback"


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
