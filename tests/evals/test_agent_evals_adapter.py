from tests.evals.evaluators.agent_evals_adapter import evaluate_with_agent_evals_or_skip


def test_agent_evals_adapter_skips_without_optional_dependency(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = evaluate_with_agent_evals_or_skip(
        [{"event": "initialization"}],
        rule_evaluation={"trajectory_score": 1.0, "failed_rules": []},
    )

    assert result["status"] == "skipped"
    assert result["rule_evaluation"]["trajectory_score"] == 1.0
