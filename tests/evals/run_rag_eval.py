# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.evals.evaluators.deepseek_judge import create_ragas_llm, has_eval_judge_key
from tests.evals.evaluators.rag_metrics import (
    RagCandidate,
    aggregate_rag_metrics,
    evaluate_rag_case,
)

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "rag_cases.jsonl"


def load_rag_cases(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_rag_eval(
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    output_path: Path | None = None,
    include_ragas: bool = False,
) -> dict[str, Any]:
    results = [
        evaluate_rag_case(case, _fixture_candidates_for_case(case, top_k=top_k), top_k=top_k)
        for case in cases
    ]
    summary = {
        "total": len(results),
        "top_k": top_k,
        "metrics": aggregate_rag_metrics(results)["metrics"],
        "ragas": _ragas_summary(cases, results) if include_ragas else None,
        "cases": results,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def build_ragas_samples(
    cases: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_by_case_id = {result["case_id"]: result for result in case_results}
    return [
        _ragas_sample_from_case(case, results_by_case_id[case["case_id"]])
        for case in cases
    ]


def _fixture_candidates_for_case(case: dict[str, Any], *, top_k: int) -> list[RagCandidate]:
    expected_ids = [str(item) for item in case.get("expected_question_ids", [])]
    negative_ids = [str(item) for item in case.get("negative_question_ids", [])]
    skill_areas = [str(item) for item in case.get("acceptable_skill_areas", [])]
    candidates = [
        RagCandidate(
            id=question_id,
            score=1.0 - (index * 0.05),
            skill_areas=tuple(skill_areas[: max(1, min(len(skill_areas), 2))]),
        )
        for index, question_id in enumerate(expected_ids)
    ]
    filler_count = max(0, top_k - len(candidates))
    candidates.extend(
        RagCandidate(
            id=f"{case['case_id']}-fixture-filler-{index + 1}",
            score=0.5 - (index * 0.01),
            skill_areas=tuple(skill_areas[-1:] or ["general"]),
        )
        for index in range(filler_count)
    )
    candidates.extend(
        RagCandidate(
            id=negative_id,
            score=0.1 - (index * 0.01),
            skill_areas=("negative-fixture",),
        )
        for index, negative_id in enumerate(negative_ids)
    )
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic RAG eval metrics.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-ragas", action="store_true")
    return parser.parse_args()


def _ragas_summary(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if importlib.util.find_spec("ragas") is None:
        return {
            "status": "skipped",
            "skippedReason": "ragas is not installed.",
        }
    if not has_eval_judge_key():
        return {
            "status": "skipped",
            "skippedReason": "No eval model API key is configured.",
        }
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics.collections.context_precision import ContextPrecision
        from ragas.metrics.collections.context_recall import ContextRecall
        from ragas.metrics.collections.context_relevance import ContextRelevance
    except ImportError as exc:
        return {
            "status": "skipped",
            "skippedReason": f"Ragas metric import failed: {exc}",
        }

    llm = create_ragas_llm()
    evaluation = evaluate(
        EvaluationDataset.from_list(build_ragas_samples(cases, results)),
        metrics=[
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            ContextRelevance(llm=llm),
        ],
        llm=llm,
        show_progress=False,
        raise_exceptions=False,
    )
    return _ragas_result_summary(evaluation, results)


def _ragas_sample_from_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    selected_ids = [str(item) for item in result.get("selected_candidate_ids", [])]
    expected_ids = [str(item) for item in case.get("expected_question_ids", [])]
    return {
        "user_input": str(case.get("query", "")),
        "retrieved_contexts": [
            _context_for_candidate_id(candidate_id, case)
            for candidate_id in selected_ids
        ],
        "retrieved_context_ids": selected_ids,
        "reference_contexts": [
            _context_for_candidate_id(candidate_id, case)
            for candidate_id in expected_ids
        ],
        "reference_context_ids": expected_ids,
        "response": _rag_response_text(result),
        "reference": _rag_reference_text(case),
    }


def _context_for_candidate_id(candidate_id: str, case: dict[str, Any]) -> str:
    skill_areas = ", ".join(str(item) for item in case.get("acceptable_skill_areas", []))
    return (
        f"question_id={candidate_id}; "
        f"round_type={case.get('round_type')}; "
        f"skill_areas={skill_areas or 'general'}"
    )


def _rag_response_text(result: dict[str, Any]) -> str:
    hit_explanation = result.get("hit_explanation") or {}
    matched = ", ".join(str(item) for item in hit_explanation.get("matched_expected_ids", []))
    leaked = ", ".join(str(item) for item in hit_explanation.get("leaked_negative_ids", []))
    return (
        f"Selected candidates: {', '.join(result.get('selected_candidate_ids', []))}. "
        f"Matched expected ids: {matched or 'none'}. "
        f"Leaked negative ids: {leaked or 'none'}."
    )


def _rag_reference_text(case: dict[str, Any]) -> str:
    return (
        "Expected retrieval should prioritize ids "
        f"{', '.join(str(item) for item in case.get('expected_question_ids', []))} "
        "and cover skill areas "
        f"{', '.join(str(item) for item in case.get('acceptable_skill_areas', []))}."
    )


def _ragas_result_summary(evaluation: Any, results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = _ragas_scores(evaluation)
    return {
        "status": "passed",
        "case_count": len(results),
        "metrics": {
            key: round(float(value), 4)
            for key, value in scores.items()
            if isinstance(value, int | float)
        },
        "case_ids": [result["case_id"] for result in results],
    }


def _ragas_scores(evaluation: Any) -> dict[str, Any]:
    if hasattr(evaluation, "scores"):
        return dict(evaluation.scores)
    if hasattr(evaluation, "to_pandas"):
        dataframe = evaluation.to_pandas()
        numeric_columns = dataframe.select_dtypes(include="number")
        return {
            column: float(numeric_columns[column].mean())
            for column in numeric_columns.columns
        }
    if isinstance(evaluation, dict):
        return evaluation
    return {}


def main() -> int:
    args = _parse_args()
    cases = load_rag_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    summary = run_rag_eval(
        cases,
        top_k=args.top_k,
        output_path=args.output,
        include_ragas=args.include_ragas,
    )
    if not args.output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
