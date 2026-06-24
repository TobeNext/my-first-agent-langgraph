import os
from pathlib import Path

from tests.evals.run_langsmith_interview_eval import (
    langsmith_inputs_from_case,
    langsmith_outputs_from_result,
    run_langsmith_eval,
    safe_langsmith_metadata,
)


def _case() -> dict:
    return {
        "case_id": "case-a",
        "redaction_version": "v1",
        "source_type": "synthetic",
        "settings": {"enableHistoricalMemory": False},
        "turns": [{"role": "user", "content": "redacted"}],
        "expected_stage_path": ["initialization", "wrap-up"],
        "expected_required_skills": ["FastAPI"],
        "must_not_claim": ["raw body"],
    }


def test_langsmith_metadata_contains_only_safe_fields(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_DATA_MODE", "redacted")

    metadata = safe_langsmith_metadata(_case())

    assert metadata == {
        "case_id": "case-a",
        "provider": "python",
        "runtime": "langgraph",
        "redaction_version": "v1",
        "source_type": "synthetic",
        "data_mode": "redacted",
    }


def test_langsmith_inputs_exclude_resume_jd_and_answer_bodies() -> None:
    inputs = langsmith_inputs_from_case(_case())

    assert inputs["case_id"] == "case-a"
    assert "resume_markdown" not in inputs
    assert "job_description_markdown" not in inputs
    assert "turns" not in inputs


def test_langsmith_outputs_keep_summary_shape_only() -> None:
    outputs = langsmith_outputs_from_result(
        {
            "case_id": "case-a",
            "final_snapshot": {
                "phase": "completed",
                "finalReportReady": True,
                "progress": {"currentStage": "completed"},
            },
            "report_status": "succeeded",
            "report_markdown_available": True,
            "trajectory_evaluation": {"trajectory_score": 1.0, "failed_rules": []},
            "agent_evals": {"status": "skipped"},
            "duration_seconds": 0.1,
            "redaction_version": "v1",
            "errors": [],
        }
    )

    assert outputs["phase"] == "completed"
    assert outputs["report_status"] == "succeeded"
    assert outputs["agent_evals"] == {"status": "skipped"}
    assert "assistant_reply" not in outputs
    assert "report_markdown" not in outputs


def test_langsmith_runner_skips_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    output_path = tmp_path / "langsmith-summary.json"

    summary = run_langsmith_eval([_case()], output_path=output_path)

    assert summary["status"] == "skipped"
    assert summary["skippedReason"] == "LANGSMITH_API_KEY is not set."
    assert output_path.exists()
    assert "LANGSMITH_API_KEY" not in os.environ
