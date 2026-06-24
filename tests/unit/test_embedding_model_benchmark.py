"""Tests for the embedding model benchmark script."""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.evals.run_embedding_model_benchmark import (
    EMBEDDING_MODELS,
    _build_passage_texts,
    _build_query_texts,
    _mrr,
    _ndcg,
    _negative_exclusion,
    _precision_at_k,
    _recall_at_k,
    aggregate_case_metrics,
    compute_metrics_for_case,
    cosine_similarity,
    load_eval_cases,
    load_question_bank,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def test_load_question_bank_returns_non_empty_list() -> None:
    questions = load_question_bank()
    assert isinstance(questions, list)
    assert len(questions) >= 80, f"Expected >=80 questions, got {len(questions)}"
    for item in questions:
        assert "id" in item, f"Missing id in {item}"
        assert "text" in item, f"Missing text in {item}"
        assert "skill_areas" in item, f"Missing skill_areas in {item}"


def test_load_eval_cases_returns_non_empty_list() -> None:
    cases = load_eval_cases()
    assert isinstance(cases, list)
    assert len(cases) >= 20, f"Expected >=20 cases, got {len(cases)}"
    for case in cases:
        assert "case_id" in case, f"Missing case_id in {case}"
        assert "query" in case, f"Missing query in {case}"
        assert "expected_question_ids" in case, f"Missing expected_question_ids in {case}"


def test_all_expected_ids_exist_in_question_bank() -> None:
    """Ensure that every expected_question_id in eval cases exists in the question bank."""
    questions = load_question_bank()
    question_ids = {item["id"] for item in questions}
    cases = load_eval_cases()
    for case in cases:
        for expected_id in case["expected_question_ids"]:
            assert expected_id in question_ids, (
                f"Case '{case['case_id']}' references unknown question '{expected_id}'"
            )


def test_all_negative_ids_exist_in_question_bank() -> None:
    """Ensure that every negative_question_id in eval cases exists in the question bank."""
    questions = load_question_bank()
    question_ids = {item["id"] for item in questions}
    cases = load_eval_cases()
    for case in cases:
        for neg_id in case.get("negative_question_ids", []):
            assert neg_id in question_ids, (
                f"Case '{case['case_id']}' references unknown negative question '{neg_id}'"
            )


# ---------------------------------------------------------------------------
# Text building (prefix/instruction)
# ---------------------------------------------------------------------------


def test_e5_build_passage_texts_adds_prefix() -> None:
    bank = [{"text": "Explain FastAPI"}]
    definition = EMBEDDING_MODELS["multilingual-e5-base"]
    texts = _build_passage_texts(bank, definition)
    assert texts == ["passage: Explain FastAPI"]


def test_e5_build_query_texts_adds_prefix() -> None:
    queries = ["What is FastAPI?"]
    definition = EMBEDDING_MODELS["multilingual-e5-base"]
    texts = _build_query_texts(queries, definition)
    assert texts == ["query: What is FastAPI?"]


def test_bge_m3_build_query_texts_adds_instruction() -> None:
    queries = ["What is RAG?"]
    definition = EMBEDDING_MODELS["bge-m3"]
    texts = _build_query_texts(queries, definition)
    assert texts[0].startswith("Represent this sentence for searching relevant passages: ")


def test_bge_large_zh_build_query_texts_adds_chinese_instruction() -> None:
    queries = ["什么是 RAG？"]
    definition = EMBEDDING_MODELS["bge-large-zh"]
    texts = _build_query_texts(queries, definition)
    assert texts[0].startswith("为这个句子生成表示以用于检索相关文章：")


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    query = np.array([1.0, 0.0, 0.0])
    docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sims = cosine_similarity(query, docs)
    assert abs(sims[0] - 1.0) < 0.001
    assert abs(sims[1]) < 0.001
    assert abs(sims[2]) < 0.001


def test_cosine_similarity_orthogonal() -> None:
    query = np.array([0.0, 1.0, 0.0])
    docs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sims = cosine_similarity(query, docs)
    assert all(abs(s) < 0.001 for s in sims)


def test_cosine_similarity_normalized() -> None:
    query = np.array([3.0, 4.0])  # length 5
    docs = np.array([[6.0, 8.0], [0.0, 1.0]])  # first: length 10, collinear
    sims = cosine_similarity(query, docs)
    assert abs(sims[0] - 1.0) < 0.001  # collinear
    assert abs(sims[1] - 0.8) < 0.001  # dot product normalized


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_recall_at_k_full() -> None:
    assert _recall_at_k(["a", "b", "c"], {"a", "b"}) == 1.0


def test_recall_at_k_partial() -> None:
    assert _recall_at_k(["a", "c", "d"], {"a", "b"}) == 0.5


def test_recall_at_k_empty_expected() -> None:
    assert _recall_at_k(["a", "b"], set()) == 0.0


def test_mrr_first_position() -> None:
    assert _mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_third_position() -> None:
    assert abs(_mrr(["a", "b", "c"], {"c"}) - (1.0 / 3)) < 0.001


def test_mrr_not_found() -> None:
    assert _mrr(["a", "b"], {"c"}) == 0.0


def test_ndcg_perfect() -> None:
    assert _ndcg(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_ndcg_partial() -> None:
    result = _ndcg(["a", "c", "d"], ["a", "b"])
    assert 0 < result < 1.0


def test_ndcg_empty_expected() -> None:
    assert _ndcg(["a", "b"], []) == 0.0


def test_precision_at_k_full() -> None:
    assert _precision_at_k(["a", "b"], {"a", "b"}) == 1.0


def test_precision_at_k_partial() -> None:
    assert _precision_at_k(["a", "b"], {"a", "c"}) == 0.5


def test_negative_exclusion_no_leaks() -> None:
    assert _negative_exclusion(["a", "b"], {"c", "d"}) == 1.0


def test_negative_exclusion_one_leak() -> None:
    assert _negative_exclusion(["a", "b", "c"], {"c", "d"}) == 0.5


def test_negative_exclusion_empty_negatives() -> None:
    assert _negative_exclusion(["a", "b"], set()) == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_case_metrics_averages_correctly() -> None:
    cases = [
        {
            "case_id": "c1",
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "ndcg_at_10": 1.0,
            "precision_at_5": 0.8,
            "negative_exclusion": 1.0,
        },
        {
            "case_id": "c2",
            "recall_at_5": 0.0,
            "recall_at_10": 0.5,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
            "precision_at_5": 0.0,
            "negative_exclusion": 1.0,
        },
    ]
    metrics = aggregate_case_metrics(cases)
    assert metrics["recall_at_5"] == 0.5
    assert metrics["recall_at_10"] == 0.75
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_10"] == 0.5
    assert metrics["precision_at_5"] == 0.4
    assert metrics["negative_exclusion"] == 1.0


# ---------------------------------------------------------------------------
# End-to-end compute_metrics_for_case (with random embeddings)
# ---------------------------------------------------------------------------


def test_compute_metrics_for_case_perfect_retrieval() -> None:
    """When query matches one doc perfectly & negatives exist, negatives should not rank above
    irrelevant docs that happen to have zero similarity (ties break by index)."""

    dim = 16
    # Create more docs than top_k so negatives can fall outside top_k.
    question_bank = [
        {"id": f"q{i}", "text": f"Question {i}"} for i in range(12)
    ]
    # Identity: q0 aligns with col 0, q3 with col 3, etc.
    doc_vecs = np.eye(len(question_bank), dim)
    query_vec = doc_vecs[0]  # matches q0 perfectly

    case: dict[str, Any] = {
        "case_id": "test-perfect",
        "expected_question_ids": ["q0"],
        "negative_question_ids": ["q3", "q4"],
    }
    result = compute_metrics_for_case(query_vec, doc_vecs, question_bank, case, top_k=10)

    assert result["recall_at_10"] == 1.0
    assert result["mrr"] == 1.0
    assert result["matched_expected"] == ["q0"]


def test_compute_metrics_for_case_miss() -> None:
    """Query is orthogonal to q0 but aligns with several other docs, pushing q0 out of top_k."""
    dim = 8
    question_bank = [{"id": f"q{i}", "text": f"Q{i}"} for i in range(12)]
    doc_vecs = np.eye(len(question_bank), dim)
    # Query matches docs 2,3,4,5,6,7 (cols 2-7), but NOT q0 (col 0) or q1 (col 1)
    raw = np.zeros(dim)
    raw[2:8] = 1.0
    query_vec = raw / np.linalg.norm(raw)

    case: dict[str, Any] = {
        "case_id": "test-miss",
        "expected_question_ids": ["q0"],
        "negative_question_ids": [],
    }
    result = compute_metrics_for_case(query_vec, doc_vecs, question_bank, case, top_k=5)
    assert result["recall_at_5"] == 0.0
    assert "q0" in result["missed_expected"]


def test_compute_metrics_negative_leak_detection() -> None:
    """Query q0 matches doc q0 but also has high similarity to q3 (negative)."""
    dim = 4
    question_bank = [{"id": f"q{i}", "text": f"Q{i}"} for i in range(4)]
    # q0 matches docs 0 and 3 (negative)
    doc_vecs = np.eye(4, dim)
    query_vec = np.array([0.5, 0.0, 0.0, 0.5])
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    case = {
        "case_id": "test-leak",
        "expected_question_ids": ["q0"],
        "negative_question_ids": ["q3"],
    }
    result = compute_metrics_for_case(query_vec, doc_vecs, question_bank, case, top_k=10)
    # q0 and q3 should both appear in top results
    assert "q3" in result["leaked_negative"]
    assert result["negative_exclusion"] == 0.0


# ---------------------------------------------------------------------------
# Model registry validation
# ---------------------------------------------------------------------------


def test_all_models_have_required_fields() -> None:
    required = {"name", "dimension", "description"}
    for key, info in EMBEDDING_MODELS.items():
        missing = required - set(info.keys())
        assert not missing, f"Model '{key}' missing fields: {missing}"


def test_e5_models_have_query_passage_prefix() -> None:
    for key in ["multilingual-e5-large", "multilingual-e5-base", "multilingual-e5-small"]:
        info = EMBEDDING_MODELS[key]
        assert info["query_prefix"] == "query: ", f"{key} missing query_prefix"
        assert info["passage_prefix"] == "passage: ", f"{key} missing passage_prefix"


def test_bge_models_have_query_instruction() -> None:
    for key in ["bge-m3", "bge-large-zh"]:
        info = EMBEDDING_MODELS[key]
        assert "query_instruction" in info, f"{key} missing query_instruction"
        assert len(info["query_instruction"]) > 0, f"{key} query_instruction is empty"
