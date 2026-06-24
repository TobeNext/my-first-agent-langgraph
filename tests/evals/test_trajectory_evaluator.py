from pathlib import Path

from tests.evals.evaluators.trajectory import evaluate_trajectory
from tests.evals.run_interview_eval_target import load_interview_cases, run_cases


def test_trajectory_evaluator_scores_valid_short_flow(tmp_path: Path) -> None:
    case = load_interview_cases()[0]
    summary = run_cases([case], include_trajectory=True, work_dir=tmp_path)
    result = summary["results"][0]

    evaluation = evaluate_trajectory(
        result["trajectory"],
        expected_stage_path=case["expected_stage_path"],
    )

    assert evaluation["trajectory_score"] == 1.0
    assert evaluation["failed_rules"] == []


def test_trajectory_evaluator_flags_bad_path() -> None:
    evaluation = evaluate_trajectory(
        [
            {
                "event": "user_turn_1",
                "phase": "professional-skills-round",
                "active_round_type": "professional-skills",
                "current_stage": "main-question",
                "final_report_ready": False,
                "report_status": None,
            }
        ],
        expected_stage_path=["initialization", "wrap-up"],
    )

    assert "starts_with_initialization" in evaluation["failed_rules"]
    assert "expected_phase:wrap-up" in evaluation["failed_rules"]
    assert evaluation["trajectory_score"] < 1.0


def test_trajectory_evaluator_rejects_body_text_fields() -> None:
    evaluation = evaluate_trajectory(
        [
            {
                "event": "initialization",
                "phase": "professional-skills-round",
                "active_round_type": "professional-skills",
                "current_stage": "main-question",
                "final_report_ready": False,
                "report_status": None,
                "candidateAnswer": "raw answer must not be included",
            }
        ]
    )

    assert "no_body_text_fields" in evaluation["failed_rules"]
