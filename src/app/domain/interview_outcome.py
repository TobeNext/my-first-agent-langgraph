from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.domain.question_generator import GeneratedQuestionRecord
from app.domain.question_retriever import RagRecallTrace
from app.schemas.interview_state import (
    AnswerAttemptState,
    InterviewSessionState,
    InterviewTopicNodeState,
)

INTERVIEW_OUTCOME_SCHEMA_VERSION = 3


def create_interview_outcome_snapshot(
    *,
    thread_id: str,
    state: InterviewSessionState,
    recall_traces: list[RagRecallTrace],
    generation_trace: list[GeneratedQuestionRecord] | None = None,
    outcome_root: str | Path | None = None,
) -> str:
    created_at = _now()
    root = _outcome_root(outcome_root)
    outcome_path = (
        root / f"{_sanitize_timestamp(created_at)}-{thread_id}" / "interview-outcome.json"
    )
    record = _build_outcome_record(
        created_at=created_at,
        updated_at=created_at,
        state=state,
        recall_traces=recall_traces,
        generation_trace=generation_trace or [],
        feedback=_pending_feedback(),
    )
    _write_json(outcome_path, record)
    _write_index(
        root=root,
        thread_id=thread_id,
        created_at=created_at,
        updated_at=created_at,
        outcome_file_path=str(outcome_path),
    )
    return str(outcome_path)


def update_interview_outcome_snapshot(
    *,
    file_path: str | Path,
    state: InterviewSessionState,
    recall_traces: list[RagRecallTrace] | None = None,
    generation_trace: list[GeneratedQuestionRecord] | None = None,
) -> None:
    path = Path(file_path)
    current = json.loads(path.read_text(encoding="utf-8"))
    updated_at = _now()
    current_selector_training = current.get("selectorTraining", {})
    record = _build_outcome_record(
        created_at=current["createdAt"],
        updated_at=updated_at,
        state=state,
        recall_traces=recall_traces
        if recall_traces is not None
        else current_selector_training.get("traces", []),
        generation_trace=generation_trace
        if generation_trace is not None
        else current_selector_training.get("generationTrace", []),
        feedback=_existing_feedback(current),
    )
    _write_json(path, record)
    _write_index(
        root=path.parents[1],
        thread_id=state.threadId,
        created_at=current["createdAt"],
        updated_at=updated_at,
        outcome_file_path=str(path),
    )


def _build_outcome_record(
    *,
    created_at: str,
    updated_at: str,
    state: InterviewSessionState,
    recall_traces: list[RagRecallTrace] | list[dict[str, Any]],
    generation_trace: list[GeneratedQuestionRecord] | list[dict[str, Any]],
    feedback: dict[str, Any],
) -> dict[str, Any]:
    generation_payload = [_serialize(item) for item in generation_trace]
    generation_map = {
        _normalize_question_text(item["questionText"]): item for item in generation_payload
    }
    recall_payload = [_serialize(item) for item in recall_traces]
    return {
        "schemaVersion": INTERVIEW_OUTCOME_SCHEMA_VERSION,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "threadId": state.threadId,
        "session": {
            "targetRole": state.targetRole,
            "responseLanguage": state.responseLanguage,
            "phase": state.phase,
            "finalReportReady": state.finalReportReady,
            "setup": state.setup.model_dump(),
            "resumeContext": state.resumeContext.model_dump(),
        },
        "selectorTraining": {
            "traces": recall_payload,
            "generationTrace": generation_payload,
            "recallEvents": _selector_training_events(recall_payload, state, generation_map),
            "selectedQuestionLabels": _selector_training_labels(
                recall_payload,
                state,
                generation_map,
            ),
        },
        "candidateImprovement": {
            **_performance_summary(state, generation_map),
            "strongSignals": _candidate_theme_records(state, "strength")[:8],
            "knowledgeWeaknesses": _knowledge_weakness_records(state)[:12],
            "questionReviews": _question_reviews(state, generation_map),
            "report": {
                "finalReport": state.finalReport,
                "lastCorrectionSummary": state.lastCorrectionSummary,
            },
            "feedback": feedback,
        },
    }


def _selector_training_events(
    traces: list[dict[str, Any]],
    state: InterviewSessionState,
    generation_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    performance_map = _question_performance_map(state, generation_map)
    events = []
    for trace in traces:
        candidates = []
        for candidate in trace.get("candidates", []):
            candidates.append(
                {
                    "questionId": candidate["id"],
                    "questionText": candidate["questionText"],
                    "vectorScore": candidate["vectorScore"],
                    "bm25Score": candidate["bm25Score"],
                    "hybridScore": candidate["hybridScore"],
                    "rerankRank": candidate["rerankRank"],
                    "finalSelectionRank": candidate["finalSelectionRank"],
                    "filterReason": candidate["filterReason"],
                    "wasSelected": candidate["finalSelectionRank"] is not None,
                    "outcomeLabel": performance_map.get(
                        _normalize_question_text(candidate["questionText"])
                    ),
                }
            )
        events.append(
            {
                "traceTimestamp": trace["timestamp"],
                "roundType": trace["roundType"],
                "skill": trace["skill"],
                "queryText": trace["queryText"],
                "logContext": trace["logContext"],
                "candidates": candidates,
            }
        )
    return events


def _selector_training_labels(
    traces: list[dict[str, Any]],
    state: InterviewSessionState,
    generation_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    performance_map = _question_performance_map(state, generation_map)
    labels = []
    for trace in traces:
        for selection in trace.get("finalSelectedQuestions", []):
            performance = performance_map.get(_normalize_question_text(selection["questionText"]))
            labels.append(
                {
                    "traceTimestamp": trace["timestamp"],
                    "roundType": trace["roundType"],
                    "skill": trace["skill"],
                    "queryText": trace["queryText"],
                    "logContext": trace["logContext"],
                    "questionId": selection["id"],
                    "questionText": selection["questionText"],
                    "vectorScore": selection["vectorScore"],
                    "bm25Score": selection["bm25Score"],
                    "hybridScore": selection["hybridScore"],
                    "rerankRank": selection["rerankRank"],
                    "finalSelectionRank": selection["finalSelectionRank"],
                    "questionDriver": (performance or {}).get("questionDriver", "resume"),
                    "resumeSignals": (performance or {}).get("resumeSignals", []),
                    "jobDescriptionSignals": (performance or {}).get(
                        "jobDescriptionSignals",
                        [],
                    ),
                    "selectionReason": (performance or {}).get(
                        "selectionReason",
                        "No performance label was available for this selection.",
                    ),
                    "performance": performance,
                }
            )
    return labels


def _performance_summary(
    state: InterviewSessionState,
    generation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_nodes = [node for round_item in state.rounds for node in round_item.nodes]
    scored = [
        node.aggregatedScore for node in all_nodes if isinstance(node.aggregatedScore, int | float)
    ]
    completed = [node for node in all_nodes if node.status in {"completed", "skipped"}]
    return {
        "totalQuestionCount": len(all_nodes),
        "completedQuestionCount": len(completed),
        "finalScore": _average(scored),
        "rounds": [
            {
                "id": round_item.id,
                "type": round_item.type,
                "status": round_item.status,
                "plannedNodeCount": round_item.plannedNodeCount,
                "completedNodeCount": round_item.completedNodeCount,
                "activeNodeId": round_item.activeNodeId,
                "nodes": [_node_record(node, generation_map) for node in round_item.nodes],
            }
            for round_item in state.rounds
        ],
    }


def _node_record(
    node: InterviewTopicNodeState,
    generation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trace = _resolve_question_trace(node, generation_map)
    return {
        "id": node.id,
        "topic": node.topic,
        "source": node.source,
        "mainQuestion": node.mainQuestion,
        "questionDriver": trace["questionDriver"],
        "resumeSignals": trace["resumeSignals"],
        "jobDescriptionSignals": trace["jobDescriptionSignals"],
        "selectionReason": trace["selectionReason"],
        "status": node.status,
        "aggregatedScore": _round_number(node.aggregatedScore),
        "followUpCount": node.followUpCount,
        "maxFollowUps": node.maxFollowUps,
        "earlyCompletionReason": node.earlyCompletionReason,
        "summary": node.summary.model_dump() if node.summary else None,
        "answerAttempts": [_attempt_record(attempt) for attempt in node.answerAttempts],
    }


def _attempt_record(attempt: AnswerAttemptState) -> dict[str, Any]:
    return {
        "id": attempt.id,
        "targetType": attempt.targetType,
        "targetId": attempt.targetId,
        "classification": attempt.classification,
        "createdAt": attempt.createdAt,
        "userMessage": attempt.userMessage,
        "score": attempt.score.model_dump() if attempt.score else None,
        "strengths": attempt.strengths,
        "missingPoints": attempt.missingPoints,
        "incorrectPoints": attempt.incorrectPoints,
        "isDetour": attempt.isDetour,
    }


def _question_performance_map(
    state: InterviewSessionState,
    generation_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        _normalize_question_text(node.mainQuestion): _question_performance(
            round_item.type,
            node,
            generation_map,
        )
        for round_item in state.rounds
        for node in round_item.nodes
    }


def _question_performance(
    round_type: str,
    node: InterviewTopicNodeState,
    generation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    latest = node.answerAttempts[-1] if node.answerAttempts else None
    trace = _resolve_question_trace(node, generation_map)
    return {
        "roundType": round_type,
        "topic": node.topic,
        "mainQuestion": node.mainQuestion,
        "questionDriver": trace["questionDriver"],
        "resumeSignals": trace["resumeSignals"],
        "jobDescriptionSignals": trace["jobDescriptionSignals"],
        "selectionReason": trace["selectionReason"],
        "nodeStatus": node.status,
        "aggregatedScore": _round_number(node.aggregatedScore),
        "latestClassification": latest.classification if latest else None,
        "strengths": node.summary.strengths if node.summary else [],
        "missingPoints": node.summary.missingPoints if node.summary else [],
        "incorrectPoints": node.summary.weaknesses if node.summary else [],
    }


def _candidate_theme_records(state: InterviewSessionState, kind: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for round_item in state.rounds:
        for node in round_item.nodes:
            values = _summary_values(node, kind)
            for value in values:
                key = _normalize_question_text(value).lower()
                bucket = buckets.setdefault(
                    key,
                    {
                        "theme": value,
                        "frequency": 0,
                        "affectedTopics": set(),
                        "exampleQuestions": set(),
                        "evidence": set(),
                        "scores": [],
                    },
                )
                bucket["frequency"] += 1
                bucket["affectedTopics"].add(node.topic)
                bucket["exampleQuestions"].add(node.mainQuestion)
                if node.summary:
                    bucket["evidence"].update(node.summary.evidence)
                if isinstance(node.aggregatedScore, int | float):
                    bucket["scores"].append(node.aggregatedScore)
    return [
        {
            "theme": bucket["theme"],
            "frequency": bucket["frequency"],
            "affectedTopics": list(bucket["affectedTopics"]),
            "exampleQuestions": list(bucket["exampleQuestions"])[:3],
            "evidence": list(bucket["evidence"])[:3],
            "averageQuestionScore": _average(bucket["scores"]),
        }
        for bucket in buckets.values()
    ]


def _knowledge_weakness_records(state: InterviewSessionState) -> list[dict[str, Any]]:
    records = []
    for kind, weakness_kind in [
        ("missing", "missing-knowledge"),
        ("incorrect", "incorrect-knowledge"),
    ]:
        for theme in _candidate_theme_records(state, kind):
            records.append(
                {
                    **theme,
                    "kind": weakness_kind,
                    "priority": _weakness_priority(
                        theme["frequency"],
                        theme["averageQuestionScore"],
                    ),
                }
            )
    return records


def _question_reviews(
    state: InterviewSessionState,
    generation_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    reviews = []
    for round_item in state.rounds:
        for node in round_item.nodes:
            trace = _resolve_question_trace(node, generation_map)
            reviews.append(
                {
                    "roundType": round_item.type,
                    "topic": node.topic,
                    "question": node.mainQuestion,
                    "questionDriver": trace["questionDriver"],
                    "resumeSignals": trace["resumeSignals"],
                    "jobDescriptionSignals": trace["jobDescriptionSignals"],
                    "selectionReason": trace["selectionReason"],
                    "score": _round_number(node.aggregatedScore),
                    "strengths": node.summary.strengths if node.summary else [],
                    "missingPoints": node.summary.missingPoints if node.summary else [],
                    "incorrectPoints": node.summary.weaknesses if node.summary else [],
                    "improvementAdvice": node.summary.improvementAdvice if node.summary else [],
                    "evidence": node.summary.evidence[:3] if node.summary else [],
                }
            )
    return reviews


def _resolve_question_trace(
    node: InterviewTopicNodeState,
    generation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return generation_map.get(_normalize_question_text(node.mainQuestion)) or {
        "questionDriver": "resume",
        "resumeSignals": [],
        "jobDescriptionSignals": [],
        "selectionReason": "No generation trace was available for this question.",
    }


def _summary_values(node: InterviewTopicNodeState, kind: str) -> list[str]:
    if not node.summary:
        return []
    if kind == "strength":
        return node.summary.strengths
    if kind == "missing":
        return node.summary.missingPoints
    return node.summary.weaknesses


def _weakness_priority(frequency: int, average_score: float | None) -> str:
    if frequency >= 2 or (average_score is not None and average_score < 6.5):
        return "high"
    if average_score is not None and average_score < 7.5:
        return "medium"
    return "low"


def _pending_feedback() -> dict[str, Any]:
    return {
        "status": "pending",
        "submittedAt": None,
        "overallExperienceScore": None,
        "questionFitScore": None,
        "difficultyScore": None,
        "comment": None,
    }


def _existing_feedback(record: dict[str, Any]) -> dict[str, Any]:
    return (
        record.get("candidateImprovement", {}).get("feedback")
        or record.get("userFeedback")
        or _pending_feedback()
    )


def _write_index(
    *,
    root: Path,
    thread_id: str,
    created_at: str,
    updated_at: str,
    outcome_file_path: str,
) -> None:
    _write_json(
        root / "index" / f"{thread_id}.json",
        {
            "threadId": thread_id,
            "createdAt": created_at,
            "updatedAt": updated_at,
            "outcomeFilePath": outcome_file_path,
        },
    )


def _outcome_root(root: str | Path | None) -> Path:
    path = Path(root) if root is not None else Path(get_settings().outcome_root)
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


def _round_number(value: float | None) -> float | None:
    return round(value, 4) if isinstance(value, int | float) else None


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _sanitize_timestamp(timestamp: str) -> str:
    return timestamp.replace(":", "-").replace(".", "-")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
