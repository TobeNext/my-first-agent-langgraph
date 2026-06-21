from app.domain.follow_up_memory import (
    TEXT_SEGMENT_LIMIT,
    build_follow_up_memory_snapshot,
    is_duplicate_follow_up_question,
    normalize_question_text,
)
from app.domain.interview_state_machine import get_active_node, get_active_round
from app.schemas.interview_state import HistoricalInterviewMemoryState
from tests.unit.test_interview_state_machine import _state_fixture


def test_build_follow_up_memory_snapshot_collects_asked_follow_ups_across_nodes() -> None:
    state = _state_fixture(flow_test=False)
    first_node = state.rounds[0].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[0].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "你如何判断 query rewrite 是否需要触发？",
                        "status": "asked",
                    }
                ),
                state.rounds[0].nodes[0].followUps[1].model_copy(
                    update={
                        "question": "重排阶段如何处理召回噪声？",
                        "status": "answered",
                    }
                ),
            ]
        },
        deep=True,
    )
    project_node = state.rounds[1].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[1].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "状态机如何保证恢复一致性？",
                        "status": "asked",
                    }
                )
            ]
        },
        deep=True,
    )
    state = state.model_copy(
        update={
            "rounds": [
                state.rounds[0].model_copy(update={"nodes": [first_node]}, deep=True),
                state.rounds[1].model_copy(update={"nodes": [project_node]}, deep=True),
            ]
        },
        deep=True,
    )

    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_node is not None

    snapshot = build_follow_up_memory_snapshot(state, active_node)

    assert snapshot.askedFollowUpQuestions == [
        "你如何判断 query rewrite 是否需要触发？",
        "重排阶段如何处理召回噪声？",
        "状态机如何保证恢复一致性？",
    ]


def test_build_follow_up_memory_snapshot_uses_only_current_main_question() -> None:
    state = _state_fixture(flow_test=False)
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_node is not None
    active_node = active_node.model_copy(
        update={
            "answerAttempts": [
                {
                    "id": "attempt-1",
                    "targetType": "main-question",
                    "targetId": active_node.id,
                    "userMessage": "候选人回答原文不应该进入长期追问记忆。",
                    "classification": "direct-answer",
                    "score": None,
                    "strengths": [],
                    "missingPoints": [],
                    "incorrectPoints": [],
                    "isDetour": False,
                    "createdAt": "2026-06-19T00:00:00Z",
                }
            ]
        },
        deep=True,
    )

    snapshot = build_follow_up_memory_snapshot(state, active_node)
    dumped = snapshot.model_dump_json()

    assert snapshot.currentMainQuestion == "请解释你的 RAG 链路。"
    assert "候选人回答原文" not in dumped
    assert "Current question dialogue record" not in dumped


def test_build_follow_up_memory_snapshot_normalizes_empty_job_description() -> None:
    state = _state_fixture(flow_test=False)
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_node is not None

    snapshot = build_follow_up_memory_snapshot(state, active_node)

    assert snapshot.resumeSummary.jobDescription == "not provided"


def test_build_follow_up_memory_snapshot_prefers_explicit_state_memory() -> None:
    state = _state_fixture(flow_test=False)
    state = state.model_copy(
        update={
            "followUpMemory": state.followUpMemory.model_copy(
                update={
                    "askedQuestions": ["显式记录的追问"],
                    "resumeDigest": "显式简历摘要",
                    "jobDescriptionDigest": "显式 JD 摘要",
                    "updatedAt": "2026-06-19T00:00:00Z",
                }
            )
        },
        deep=True,
    )
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_node is not None

    snapshot = build_follow_up_memory_snapshot(state, active_node)

    assert snapshot.askedFollowUpQuestions == ["显式记录的追问"]
    assert snapshot.resumeSummary.professionalSkills == "显式简历摘要"
    assert snapshot.resumeSummary.jobDescription == "显式 JD 摘要"


def test_build_follow_up_memory_snapshot_includes_historical_weakness_memory() -> None:
    state = _state_fixture(flow_test=False)
    state = state.model_copy(
        update={
            "historicalMemory": HistoricalInterviewMemoryState(
                hasMemory=True,
                sourceInterviewIds=["interview-old"],
                weaknesses=["RAG 失败降级覆盖不足"],
                missingPoints=["缺少指标阈值"],
                improvementAdvice=["补充监控和回滚策略"],
                reinforcementQuestionHints=["追问失败时如何降级"],
            )
        },
        deep=True,
    )
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_node is not None

    snapshot = build_follow_up_memory_snapshot(state, active_node)

    assert snapshot.historicalReportMemory.weaknesses == ["RAG 失败降级覆盖不足"]
    assert snapshot.historicalReportMemory.missingPoints == ["缺少指标阈值"]
    assert snapshot.historicalReportMemory.improvementAdvice == ["补充监控和回滚策略"]
    assert snapshot.historicalReportMemory.reinforcementQuestionHints == [
        "追问失败时如何降级"
    ]


def test_build_follow_up_memory_snapshot_truncates_long_resume_and_jd() -> None:
    state = _state_fixture(flow_test=False)
    long_skills = "RAG " * 500
    long_project = "Agent state machine " * 300
    long_jd = "memory orchestration " * 300
    state = state.model_copy(
        update={
            "resumeContext": state.resumeContext.model_copy(
                update={
                    "professionalSkills": long_skills,
                    "projectExperience": long_project,
                    "jobDescription": long_jd,
                }
            )
        },
        deep=True,
    )
    active_round = get_active_round(state)
    active_node = get_active_node(active_round)
    assert active_node is not None

    snapshot = build_follow_up_memory_snapshot(state, active_node)

    assert len(snapshot.resumeSummary.professionalSkills) <= TEXT_SEGMENT_LIMIT
    assert len(snapshot.resumeSummary.projectExperience) <= TEXT_SEGMENT_LIMIT
    assert len(snapshot.resumeSummary.jobDescription) <= TEXT_SEGMENT_LIMIT
    assert snapshot.resumeSummary.professionalSkills.startswith("RAG RAG")


def test_normalize_question_text_ignores_spacing_case_and_question_marks() -> None:
    assert normalize_question_text(" What is RAG ? ") == normalize_question_text("what  is rag？")


def test_is_duplicate_follow_up_question_matches_exact_and_near_duplicates() -> None:
    state = _state_fixture(flow_test=False)
    active_node = state.rounds[0].nodes[0].model_copy(
        update={
            "followUps": [
                state.rounds[0].nodes[0].followUps[0].model_copy(
                    update={
                        "question": "你如何判断 query rewrite 是否需要触发？",
                        "status": "asked",
                    }
                )
            ]
        },
        deep=True,
    )
    state = state.model_copy(
        update={
            "rounds": [
                state.rounds[0].model_copy(update={"nodes": [active_node]}, deep=True),
                state.rounds[1],
            ]
        },
        deep=True,
    )
    snapshot = build_follow_up_memory_snapshot(state, active_node)

    assert is_duplicate_follow_up_question("你如何判断 query rewrite 是否需要触发?", snapshot)
    assert is_duplicate_follow_up_question("你如何判断 query rewrite 是否需要触发的？", snapshot)
    assert not is_duplicate_follow_up_question("能展开讲讲召回候选如何重排吗？", snapshot)
