from __future__ import annotations

from typing import Any

SAFE_EVENT_KEYS = {
    "event",
    "phase",
    "active_round_type",
    "active_node_topic",
    "current_stage",
    "final_report_ready",
    "report_status",
}
BODY_LIKE_KEYS = {
    "resume_markdown",
    "resumeMarkdown",
    "job_description_markdown",
    "jobDescriptionMarkdown",
    "answer_text",
    "answerText",
    "candidateAnswer",
    "report_markdown",
    "reportMarkdown",
    "finalReport",
}


def evaluate_trajectory(
    trajectory: list[dict[str, Any]],
    *,
    expected_stage_path: list[str] | None = None,
) -> dict[str, Any]:
    failed_rules: list[str] = []
    events = [str(item.get("event") or "") for item in trajectory]
    phases = [str(item.get("phase") or "") for item in trajectory if item.get("phase")]

    if not trajectory:
        failed_rules.append("trajectory_present")
    if not events or events[0] != "initialization":
        failed_rules.append("starts_with_initialization")
    if not any(item.get("active_round_type") == "professional-skills" for item in trajectory):
        failed_rules.append("professional_round_observed")
    if any(_contains_body_like_key(item) for item in trajectory):
        failed_rules.append("no_body_text_fields")

    for expected_phase in expected_stage_path or []:
        if expected_phase != "initialization" and expected_phase not in phases:
            failed_rules.append(f"expected_phase:{expected_phase}")

    wrap_up_index = _first_index(phases, "wrap-up")
    report_index = _first_index(events, "background_report_generation")
    if wrap_up_index is not None:
        if report_index is None:
            failed_rules.append("background_report_generation_after_wrap_up")
        elif report_index < wrap_up_index:
            failed_rules.append("report_generation_order")

    final_event = trajectory[-1] if trajectory else {}
    if final_event.get("event") == "background_report_generation":
        if final_event.get("report_status") != "succeeded":
            failed_rules.append("report_status_succeeded")
        if final_event.get("final_report_ready") is not True:
            failed_rules.append("final_report_ready_after_report_generation")

    score = 1.0 if not failed_rules else max(0.0, 1 - (len(set(failed_rules)) / 8))
    return {
        "trajectory_score": round(score, 4),
        "failed_rules": sorted(set(failed_rules)),
    }


def _contains_body_like_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in BODY_LIKE_KEYS:
                return True
            if key not in SAFE_EVENT_KEYS and isinstance(child, str) and len(child) > 240:
                return True
            if _contains_body_like_key(child):
                return True
    if isinstance(value, list):
        return any(_contains_body_like_key(item) for item in value)
    return False


def _first_index(values: list[str], target: str) -> int | None:
    try:
        return values.index(target)
    except ValueError:
        return None
