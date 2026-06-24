from pathlib import Path

from tests.evals.run_interview_eval_target import load_interview_cases, run_cases

REQUIRED_SNAPSHOT_FIELDS = {
    "assistantReply",
    "phase",
    "activeRoundType",
    "activeNodeTopic",
    "finalReportReady",
    "progress",
}


def test_interview_eval_target_returns_stable_summary_shape(tmp_path: Path) -> None:
    case = load_interview_cases()[0]
    summary = run_cases([case], work_dir=tmp_path)

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 0

    result = summary["results"][0]
    assert result["case_id"] == case["case_id"]
    assert REQUIRED_SNAPSHOT_FIELDS <= result["final_snapshot"].keys()
    assert result["assistant_reply"] == result["final_snapshot"]["assistantReply"]
    assert result["redaction_version"] == case["redaction_version"]
    assert result["skipped"] is False
    assert result["errors"] == []


def test_interview_eval_target_can_complete_report_status_boundary(tmp_path: Path) -> None:
    case = load_interview_cases()[0]
    summary = run_cases([case], include_trajectory=True, work_dir=tmp_path)
    result = summary["results"][0]

    assert result["final_snapshot"]["phase"] == "completed"
    assert result["final_snapshot"]["finalReportReady"] is True
    assert result["report_status"] == "succeeded"
    assert result["report_markdown_available"] is True
    assert result["trajectory"][-1]["event"] == "background_report_generation"
    assert result["trajectory_evaluation"] == {
        "trajectory_score": 1.0,
        "failed_rules": [],
    }


def test_interview_eval_target_writes_output_file(tmp_path: Path) -> None:
    case = load_interview_cases()[0]
    output_path = tmp_path / "summary.json"

    summary = run_cases([case], output_path=output_path, work_dir=tmp_path)

    assert output_path.exists()
    assert '"total": 1' in output_path.read_text(encoding="utf-8")
    assert summary["failed"] == 0
