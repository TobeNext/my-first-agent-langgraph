import json
from pathlib import Path

from app.domain.interview_initialization_pipeline import initialize_interview_from_kickoff
from app.domain.interview_outcome import (
    create_interview_outcome_snapshot,
    update_interview_outcome_snapshot,
)
from app.domain.rag_recall_sample import (
    update_rag_recall_sample_answer_performance,
    write_initialization_rag_recall_sample,
)


class EmptyQuestionStore:
    def search(self, *, vector, top_k, round_type):
        return type("Result", (), {"questions": []})()


def _initialized(monkeypatch):
    monkeypatch.setattr(
        "app.domain.question_retriever.MilvusQuestionStore",
        lambda: EmptyQuestionStore(),
    )
    raw = json.dumps(
        {
            "requestKind": "interview-start",
            "protocolVersion": "2026-05-structured-start-v1",
            "startInterview": True,
            "threadId": "artifact-thread",
            "resumeMarkdown": "# Resume",
            "jobDescriptionMarkdown": "# 岗位要求\n- RAG 检索",
            "settings": {
                "reviewIncorrectOrMissingPoints": True,
                "skipProfessionalSkillsRound": False,
                "skipProjectExperienceRound": True,
                "enableFlowTestMode": True,
                "professionalQuestionMode": "custom-count",
                "professionalQuestionCount": 1,
                "projectQuestionCount": 0,
            },
            "resumeSections": {
                "professionalSkills": "- RAG 检索",
                "projectExperience": "- AI 面试 Agent 状态机改造",
            },
        },
        ensure_ascii=False,
    )
    return initialize_interview_from_kickoff(
        thread_id="artifact-thread",
        raw_kickoff_message=raw,
    )


def test_outcome_writer_preserves_index_and_feedback_shape(
    tmp_path: Path, monkeypatch
) -> None:
    initialized = _initialized(monkeypatch)
    outcome_path = create_interview_outcome_snapshot(
        thread_id=initialized.state.threadId,
        state=initialized.state,
        recall_traces=initialized.resources.recallTraces,
        generation_trace=initialized.resources.generationTrace,
        outcome_root=tmp_path / "Interview outcome",
    )

    index_path = tmp_path / "Interview outcome" / "index" / "artifact-thread.json"
    index_record = json.loads(index_path.read_text(encoding="utf-8"))
    record = json.loads(Path(outcome_path).read_text(encoding="utf-8"))

    assert index_record["threadId"] == "artifact-thread"
    assert index_record["outcomeFilePath"] == outcome_path
    assert record["schemaVersion"] == 3
    assert record["selectorTraining"]["traces"][0]["timestamp"]
    assert "generationTrace" in record["selectorTraining"]
    assert record["candidateImprovement"]["feedback"] == {
        "status": "pending",
        "submittedAt": None,
        "overallExperienceScore": None,
        "questionFitScore": None,
        "difficultyScore": None,
        "comment": None,
    }

    update_interview_outcome_snapshot(file_path=outcome_path, state=initialized.state)
    updated_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert updated_index["updatedAt"] >= index_record["updatedAt"]


def test_rag_recall_sample_preserves_offline_sample_shape(
    tmp_path: Path, monkeypatch
) -> None:
    initialized = _initialized(monkeypatch)
    sample_path = write_initialization_rag_recall_sample(
        thread_id=initialized.state.threadId,
        target_role=initialized.state.targetRole,
        recall_traces=initialized.resources.recallTraces,
        state=initialized.state,
        rag_log_root=tmp_path / "RAG LOG INFO",
    )

    sample = json.loads(Path(sample_path).read_text(encoding="utf-8"))

    assert sample["schemaVersion"] == 1
    assert sample["threadId"] == "artifact-thread"
    assert sample["recalls"][0]["candidateQuestionIds"] == []
    assert "postInterviewAnswerPerformance" in sample["recalls"][0]
    assert sample["interviewSnapshot"]["phase"] == initialized.state.phase
    assert sample["interviewSnapshot"]["answerPerformances"][0]["mainQuestion"]

    update_rag_recall_sample_answer_performance(sample_path, initialized.state)
    updated = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    assert updated["updatedAt"] >= sample["updatedAt"]
