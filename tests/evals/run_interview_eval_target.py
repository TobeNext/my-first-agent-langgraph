# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings
from app.graphs.interview_graph import (
    build_interview_graph,
    invoke_interview_graph,
    run_report_generation_for_thread,
    should_start_background_report_generation,
)
from app.schemas.api import MastraStreamRequest
from app.schemas.interview_snapshot import InterviewStateSnapshot
from tests.evals.evaluators.trajectory import evaluate_trajectory

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "interview_cases.jsonl"
SAFE_ENV_DEFAULTS = {
    "MODEL_PROVIDER": "mock",
    "OTEL_SDK_DISABLED": "true",
}


class EmptyQuestionStore:
    def search(self, *, vector: Any, top_k: int, round_type: str) -> Any:
        return type("Result", (), {"questions": []})()


def load_interview_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_cases(
    cases: Iterable[dict[str, Any]],
    *,
    output_path: Path | None = None,
    include_trajectory: bool = False,
    use_real_question_store: bool = False,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    if not use_real_question_store:
        _install_empty_question_store()

    started = time.perf_counter()
    managed_temp_dir = (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) if work_dir is None else None
    )
    root = work_dir or Path(managed_temp_dir.name)
    root.mkdir(parents=True, exist_ok=True)
    try:
        results = [
            _run_case(
                case,
                case_dir=root / _safe_path_name(case["case_id"]),
                include_trajectory=include_trajectory,
            )
            for case in cases
        ]
        summary = {
            "total": len(results),
            "passed": sum(1 for result in results if not result["errors"]),
            "failed": sum(1 for result in results if result["errors"]),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "results": results,
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return summary
    finally:
        if managed_temp_dir:
            managed_temp_dir.cleanup()


def _run_case(
    case: dict[str, Any],
    *,
    case_dir: Path,
    include_trajectory: bool,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    _configure_case_environment(case_dir)
    thread_id = f"eval-{case['case_id']}"
    trajectory: list[dict[str, Any]] = []
    errors: list[str] = []
    start = time.perf_counter()
    final_state: dict[str, Any] = {}
    report_state: dict[str, Any] | None = None
    context = SqliteSaver.from_conn_string(str(case_dir / "checkpoints.db"))
    saver = context.__enter__()
    try:
        graph = build_interview_graph(checkpointer=saver)
        final_state = invoke_interview_graph(
            _request(thread_id, _start_payload(case, thread_id)),
            graph=graph,
        )
        _append_trajectory(trajectory, "initialization", final_state)
        for index, turn in enumerate(case.get("turns", []), start=1):
            final_state = invoke_interview_graph(
                _request(thread_id, str(turn.get("content", ""))),
                graph=graph,
            )
            _append_trajectory(trajectory, f"user_turn_{index}", final_state)

        if should_start_background_report_generation(final_state):
            report_state = run_report_generation_for_thread(thread_id, graph=graph)
            final_state = report_state
            _append_trajectory(trajectory, "background_report_generation", final_state)
    except Exception as exc:  # pragma: no cover - exercised by CLI failure paths.
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        context.__exit__(None, None, None)

    snapshot_payload = final_state.get("snapshot") or {}
    final_snapshot = _validate_snapshot(snapshot_payload, errors)
    report_status = (
        final_state.get("report_status")
        or (report_state or {}).get("report_status")
        or ("not-started" if not report_state else "unknown")
    )
    return {
        "case_id": case["case_id"],
        "final_snapshot": final_snapshot,
        "assistant_reply": final_snapshot.get("assistantReply", ""),
        "trajectory": trajectory if include_trajectory else [],
        "trajectory_evaluation": (
            evaluate_trajectory(
                trajectory,
                expected_stage_path=case.get("expected_stage_path", []),
            )
            if include_trajectory
            else None
        ),
        "report_status": report_status,
        "report_markdown_available": bool(final_state.get("report_markdown_available", False)),
        "duration_seconds": round(time.perf_counter() - start, 3),
        "redaction_version": case["redaction_version"],
        "skipped": False,
        "errors": errors,
    }


def _configure_case_environment(case_dir: Path) -> None:
    for key, value in SAFE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ["REPORT_DATABASE_URL"] = f"sqlite:///{case_dir / 'interview_reports.db'}"
    os.environ["OUTCOME_ROOT"] = str(case_dir / "Interview outcome")
    os.environ["RAG_LOG_ROOT"] = str(case_dir / "RAG LOG INFO")
    os.environ.pop("INTERVIEW_MEMORY_USER_ID", None)
    get_settings.cache_clear()


def _request(thread_id: str, message: str) -> MastraStreamRequest:
    return MastraStreamRequest.model_validate(
        {
            "messages": [{"role": "user", "content": message}],
            "memory": {
                "thread": thread_id,
                "resource": f"frontend-interview-{thread_id}",
            },
            "maxSteps": 5,
        }
    )


def _start_payload(case: dict[str, Any], thread_id: str) -> str:
    return json.dumps(
        {
            "requestKind": "interview-start",
            "protocolVersion": "2026-05-structured-start-v1",
            "startInterview": True,
            "threadId": thread_id,
            "resumeMarkdown": case["resume_markdown"],
            "jobDescriptionMarkdown": case["job_description_markdown"],
            "settings": case["settings"],
            "resumeSections": _resume_sections(case["resume_markdown"]),
        },
        ensure_ascii=False,
    )


def _resume_sections(resume_markdown: str) -> dict[str, str]:
    return {
        "professionalSkills": _section_after_marker(resume_markdown, "Skills"),
        "projectExperience": _section_after_marker(resume_markdown, "Projects"),
    }


def _section_after_marker(markdown: str, marker: str) -> str:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if marker.lower() in line.lower():
            section_lines = []
            for child in lines[index + 1 :]:
                if child.startswith("#"):
                    break
                section_lines.append(child)
            return "\n".join(section_lines).strip() or markdown
    return markdown


def _append_trajectory(
    trajectory: list[dict[str, Any]],
    event: str,
    state: dict[str, Any],
) -> None:
    snapshot = state.get("snapshot") or {}
    progress = snapshot.get("progress") or {}
    trajectory.append(
        {
            "event": event,
            "phase": snapshot.get("phase"),
            "active_round_type": snapshot.get("activeRoundType"),
            "active_node_topic": snapshot.get("activeNodeTopic"),
            "current_stage": progress.get("currentStage"),
            "final_report_ready": snapshot.get("finalReportReady"),
            "report_status": state.get("report_status"),
        }
    )


def _validate_snapshot(snapshot_payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if not snapshot_payload:
        errors.append("missing final snapshot")
        return {}
    try:
        return InterviewStateSnapshot.model_validate(snapshot_payload).model_dump()
    except Exception as exc:
        errors.append(f"invalid final snapshot: {type(exc).__name__}: {exc}")
        return snapshot_payload


def _install_empty_question_store() -> None:
    import app.domain.question_retriever as question_retriever

    question_retriever.MilvusQuestionStore = lambda: EmptyQuestionStore()


def _safe_path_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local deterministic interview eval cases.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-trajectory", action="store_true")
    parser.add_argument("--use-real-question-store", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = load_interview_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    summary = run_cases(
        cases,
        output_path=args.output,
        include_trajectory=args.include_trajectory,
        use_real_question_store=args.use_real_question_store,
    )
    if not args.output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
