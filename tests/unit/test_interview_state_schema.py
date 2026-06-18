from app.schemas.interview_snapshot import InterviewStateSnapshot
from app.schemas.interview_state import InterviewSessionState


def _state_fixture() -> dict:
    return {
        "version": 1,
        "threadId": "thread-1",
        "targetRole": "通用技术岗位",
        "company": None,
        "responseLanguage": "zh",
        "phase": "professional-skills-round",
        "activeRoundId": "round-professional",
        "finalReportReady": False,
        "finalReport": None,
        "setup": {
            "selectedDirection": "通用技术岗位",
            "directionSource": "derived",
            "settings": {
                "reviewIncorrectOrMissingPoints": True,
                "skipProfessionalSkillsRound": False,
                "skipProjectExperienceRound": False,
                "enableFlowTestMode": True,
                "professionalQuestionMode": "custom-count",
                "professionalQuestionCount": 1,
                "projectQuestionCount": 1,
            },
        },
        "resumeContext": {
            "professionalSkills": "TypeScript\nRAG",
            "projectExperience": "AI 面试 Agent 状态机改造",
            "jobDescription": "",
            "resumeParsed": True,
        },
        "lastCorrectionSummary": None,
        "rounds": [
            {
                "id": "round-professional",
                "type": "professional-skills",
                "status": "in-progress",
                "plannedNodeCount": 1,
                "completedNodeCount": 0,
                "activeNodeId": "node-rag",
                "nodeOrder": ["node-rag"],
                "nodes": [
                    {
                        "id": "node-rag",
                        "topic": "RAG",
                        "source": "knowledge-base",
                        "mainQuestion": "请解释你的 RAG 链路。",
                        "status": "awaiting-main-answer",
                        "currentTargetType": "main-question",
                        "currentFollowUpId": None,
                        "followUpCount": 0,
                        "maxFollowUps": 3,
                        "detourResponseCount": 0,
                        "earlyCompletionReason": None,
                        "followUps": [
                            {
                                "id": "follow-up-1",
                                "index": 1,
                                "intent": "depth",
                                "question": "",
                                "status": "pending",
                                "linkedAnswerId": None,
                            }
                        ],
                        "answerAttempts": [],
                        "aggregatedScore": None,
                        "summary": None,
                    }
                ],
            },
            {
                "id": "round-project",
                "type": "project-experience",
                "status": "pending",
                "plannedNodeCount": 1,
                "completedNodeCount": 0,
                "activeNodeId": "node-project",
                "nodeOrder": ["node-project"],
                "nodes": [
                    {
                        "id": "node-project",
                        "topic": "状态机改造",
                        "source": "resume",
                        "mainQuestion": "请介绍你的状态机项目。",
                        "status": "pending",
                        "currentTargetType": "main-question",
                        "currentFollowUpId": None,
                        "followUpCount": 0,
                        "maxFollowUps": 2,
                        "detourResponseCount": 0,
                        "earlyCompletionReason": None,
                        "followUps": [
                            {
                                "id": "project-follow-up-1",
                                "index": 1,
                                "intent": "depth",
                                "question": "",
                                "status": "pending",
                                "linkedAnswerId": None,
                            }
                        ],
                        "answerAttempts": [],
                        "aggregatedScore": None,
                        "summary": None,
                    }
                ],
            },
        ],
    }


def test_interview_session_state_accepts_ts_shape_and_dumps_same_keys() -> None:
    parsed = InterviewSessionState.model_validate(_state_fixture())
    dumped = parsed.model_dump(exclude_none=True)

    assert dumped["threadId"] == "thread-1"
    assert dumped["rounds"][0]["activeNodeId"] == "node-rag"
    assert dumped["rounds"][0]["nodes"][0]["currentTargetType"] == "main-question"


def test_interview_state_snapshot_accepts_frontend_contract_shape() -> None:
    snapshot = {
        "assistantReply": "权威回复",
        "flowTestMockUserReply": None,
        "phase": "professional-skills-round",
        "activeRoundType": "professional-skills",
        "activeNodeTopic": "RAG",
        "finalReportReady": False,
        "progress": {
            "totalQuestionCount": 2,
            "completedQuestionCount": 0,
            "remainingQuestionCount": 2,
            "currentQuestionIndex": 1,
            "currentRoundType": "professional-skills",
            "currentRoundLabel": "【第一轮：专业技能面试】",
            "currentStage": "main-question",
            "currentFollowUpIndex": None,
            "currentQuestionText": "请解释你的 RAG 链路。",
            "currentNodeTopic": "RAG",
        },
    }

    parsed = InterviewStateSnapshot.model_validate(snapshot)

    assert parsed.model_dump() == snapshot
