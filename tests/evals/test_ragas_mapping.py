from tests.evals.run_rag_eval import build_ragas_samples, load_rag_cases, run_rag_eval


def test_build_ragas_samples_maps_rag_cases_to_single_turn_shape() -> None:
    case = load_rag_cases()[0]
    rag_result = run_rag_eval([case], top_k=5)["cases"][0]

    samples = build_ragas_samples([case], [rag_result])

    assert len(samples) == 1
    sample = samples[0]
    assert sample["user_input"] == case["query"]
    assert sample["retrieved_contexts"]
    assert sample["reference_contexts"]
    assert sample["retrieved_context_ids"] == rag_result["selected_candidate_ids"]
    assert sample["reference_context_ids"] == case["expected_question_ids"]
    assert case["expected_question_ids"][0] in sample["reference"]
    assert "question_id=" in sample["retrieved_contexts"][0]
    assert "round_type=" in sample["retrieved_contexts"][0]


def test_build_ragas_samples_uses_safe_context_summaries_only() -> None:
    case = load_rag_cases()[0]
    rag_result = run_rag_eval([case], top_k=5)["cases"][0]
    sample = build_ragas_samples([case], [rag_result])[0]
    joined_context = "\n".join(sample["retrieved_contexts"])

    assert case["query"] not in joined_context
    assert "question_id=" in joined_context
    assert "skill_areas=" in joined_context


def test_run_rag_eval_marks_ragas_skipped_without_eval_key(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    summary = run_rag_eval(load_rag_cases()[:1], top_k=5, include_ragas=True)

    assert summary["ragas"] == {
        "status": "skipped",
        "skippedReason": "No eval model API key is configured.",
    }
