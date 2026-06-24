from pathlib import Path

from tests.evals.evaluators.rag_metrics import (
    RagCandidate,
    aggregate_rag_metrics,
    evaluate_rag_case,
)
from tests.evals.run_rag_eval import load_rag_cases, run_rag_eval


def _case() -> dict:
    return {
        "case_id": "rag-test",
        "expected_question_ids": ["expected-a", "expected-b"],
        "acceptable_skill_areas": ["FastAPI", "database persistence"],
        "negative_question_ids": ["negative-a"],
    }


def test_rag_metrics_scores_hits_rank_and_skill_coverage() -> None:
    result = evaluate_rag_case(
        _case(),
        [
            RagCandidate("expected-a", 0.9, ("FastAPI",)),
            RagCandidate("other", 0.8, ("database persistence",)),
            RagCandidate("expected-b", 0.7, ("FastAPI",)),
        ],
        top_k=3,
    )

    assert result["hit_rate_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg"] > 0.9
    assert result["negative_question_exclusion"] == 1.0
    assert result["skill_area_coverage"] == 1.0
    assert result["rerank_top_k_stability"] == 1.0


def test_rag_metrics_flags_negative_question_leak() -> None:
    result = evaluate_rag_case(
        _case(),
        [
            RagCandidate("negative-a", 0.95, ("unrelated",)),
            RagCandidate("other", 0.9, ("unrelated",)),
        ],
        top_k=2,
    )

    assert result["hit_rate_at_k"] == 0.0
    assert result["negative_question_exclusion"] == 0.0
    assert result["hit_explanation"]["leaked_negative_ids"] == ["negative-a"]


def test_rag_metrics_stability_drops_when_order_disagrees_with_scores() -> None:
    result = evaluate_rag_case(
        _case(),
        [
            RagCandidate("lower", 0.1, ("FastAPI",)),
            RagCandidate("higher", 0.9, ("FastAPI",)),
        ],
        top_k=2,
    )

    assert result["rerank_top_k_stability"] == 0.0


def test_aggregate_rag_metrics_averages_case_results() -> None:
    first = evaluate_rag_case(_case(), [RagCandidate("expected-a", 0.9, ("FastAPI",))], top_k=1)
    second = evaluate_rag_case(_case(), [RagCandidate("negative-a", 0.9, ("other",))], top_k=1)

    summary = aggregate_rag_metrics([first, second])

    assert summary["case_count"] == 2
    assert summary["metrics"]["negative_question_exclusion"] == 0.5


def test_run_rag_eval_outputs_summary_file(tmp_path: Path) -> None:
    output_path = tmp_path / "rag-summary.json"
    summary = run_rag_eval(load_rag_cases()[:5], top_k=5, output_path=output_path)

    assert summary["total"] == 5
    assert summary["metrics"]["hit_rate_at_k"] == 1.0
    assert summary["metrics"]["negative_question_exclusion"] == 1.0
    assert output_path.exists()


def test_run_rag_eval_can_mark_ragas_optional_metrics_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EVAL_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    summary = run_rag_eval(load_rag_cases()[:1], top_k=5, output_path=None, include_ragas=True)

    assert summary["ragas"]["status"] == "skipped"
