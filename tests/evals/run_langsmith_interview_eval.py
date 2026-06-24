# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.evals.evaluators.agent_evals_adapter import evaluate_with_agent_evals_or_skip
from tests.evals.run_interview_eval_target import load_interview_cases, run_cases

DEFAULT_DATASET_NAME = "interview-agent-eval-cases"


def safe_langsmith_metadata(case: dict[str, Any]) -> dict[str, str]:
    return {
        "case_id": str(case["case_id"]),
        "provider": "python",
        "runtime": "langgraph",
        "redaction_version": str(case["redaction_version"]),
        "source_type": str(case["source_type"]),
        "data_mode": os.environ.get("LANGSMITH_DATA_MODE", "redacted"),
    }


def langsmith_inputs_from_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "settings": case["settings"],
        "turn_count": len(case.get("turns", [])),
        "expected_stage_path": case.get("expected_stage_path", []),
        "expected_required_skills": case.get("expected_required_skills", []),
        "must_not_claim": case.get("must_not_claim", []),
        "redaction_version": case["redaction_version"],
    }


def langsmith_outputs_from_result(result: dict[str, Any]) -> dict[str, Any]:
    final_snapshot = result.get("final_snapshot") or {}
    progress = final_snapshot.get("progress") or {}
    return {
        "case_id": result["case_id"],
        "phase": final_snapshot.get("phase"),
        "final_report_ready": final_snapshot.get("finalReportReady"),
        "current_stage": progress.get("currentStage"),
        "report_status": result.get("report_status"),
        "report_markdown_available": result.get("report_markdown_available"),
        "trajectory_evaluation": result.get("trajectory_evaluation"),
        "agent_evals": result.get("agent_evals"),
        "duration_seconds": result.get("duration_seconds"),
        "redaction_version": result.get("redaction_version"),
        "errors": result.get("errors", []),
    }


def run_langsmith_eval(
    cases: list[dict[str, Any]],
    *,
    output_path: Path | None = None,
    include_trajectory: bool = False,
    include_agent_evals: bool = False,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> dict[str, Any]:
    if not os.environ.get("LANGSMITH_API_KEY"):
        summary = {
            "status": "skipped",
            "skippedReason": "LANGSMITH_API_KEY is not set.",
            "dataset_name": dataset_name,
            "case_count": len(cases),
            "results": [],
        }
        _write_summary(summary, output_path)
        return summary

    started = time.perf_counter()
    target_summary = run_cases(cases, include_trajectory=include_trajectory)
    if include_agent_evals:
        for result in target_summary["results"]:
            result["agent_evals"] = evaluate_with_agent_evals_or_skip(
                result.get("trajectory", []),
                rule_evaluation=result.get("trajectory_evaluation") or {},
            )
    experiment_id = f"local-{int(time.time())}"
    experiment_url = None

    from langsmith import Client

    client = Client()
    dataset = _get_or_create_dataset(client, dataset_name)
    dataset_id = getattr(dataset, "id", None)

    for case, result in zip(cases, target_summary["results"], strict=True):
        client.create_example(
            inputs=langsmith_inputs_from_case(case),
            outputs=langsmith_outputs_from_result(result),
            metadata=safe_langsmith_metadata(case),
            dataset_id=dataset_id,
        )

    summary = {
        "status": "passed" if target_summary["failed"] == 0 else "failed",
        "dataset_name": dataset_name,
        "dataset_id": str(dataset_id) if dataset_id else None,
        "experiment_id": experiment_id,
        "experiment_url": experiment_url,
        "case_count": len(cases),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "results": [
            {
                "case_id": result["case_id"],
                "metadata": safe_langsmith_metadata(case),
                "outputs": langsmith_outputs_from_result(result),
            }
            for case, result in zip(cases, target_summary["results"], strict=True)
        ],
    }
    _write_summary(summary, output_path)
    return summary


def _get_or_create_dataset(client: Any, dataset_name: str) -> Any:
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        return client.create_dataset(
            dataset_name=dataset_name,
            description="Redacted interview agent evaluation cases.",
        )


def _write_summary(summary: dict[str, Any], output_path: Path | None) -> None:
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangSmith interview eval mapping.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path(".tmp/langsmith-eval-summary.json"))
    parser.add_argument("--include-trajectory", action="store_true")
    parser.add_argument("--include-agent-evals", action="store_true")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = load_interview_cases(args.dataset) if args.dataset else load_interview_cases()
    if args.limit is not None:
        cases = cases[: args.limit]
    summary = run_langsmith_eval(
        cases,
        output_path=args.output,
        include_trajectory=args.include_trajectory,
        include_agent_evals=args.include_agent_evals,
        dataset_name=args.dataset_name,
    )
    if args.output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
