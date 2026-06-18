from app.domain.question_metadata import (
    build_skill_area_audit,
    clean_interview_question_metadata,
    normalize_skill_area_from_metadata,
    normalize_skill_area_from_text,
)


def test_clean_metadata_removes_legacy_fields_and_backfills_skill_area() -> None:
    cleaned = clean_interview_question_metadata(
        {
            "question": "请解释 RAG 召回后如何用 Milvus 和 BM25 rerank。",
            "answer": "Use vector search and reranking.",
            "mainCategory": "AI",
            "subCategory": "向量数据库",
            "company": "demo",
            "tags": ["RAG", "Milvus"],
            "role": "professional-skills",
            "difficulty": "hard",
        }
    )

    assert cleaned["questionType"] == "knowledge-check"
    assert cleaned["source"] == "interview-question-bank"
    assert cleaned["role"] == "professional-skills"
    assert cleaned["difficulty"] == "hard"
    assert cleaned["skillArea"] == ["rag", "milvus", "bm25"]
    assert "mainCategory" not in cleaned["metadata"]
    assert "subCategory" not in cleaned["metadata"]
    assert "company" not in cleaned["metadata"]
    assert "tags" not in cleaned["metadata"]
    assert "role" not in cleaned["metadata"]


def test_normalize_skill_area_matches_mixed_chinese_and_english_signals() -> None:
    assert normalize_skill_area_from_text(
        "TypeScript Node.js RAG 检索 Milvus 向量数据库 workflow 工具调用"
    ) == ["typescript", "rag", "milvus", "tool-calling", "workflow"]


def test_explicit_skill_area_wins_and_default_is_agent() -> None:
    assert normalize_skill_area_from_metadata({"skillArea": "RAG, Milvus tool-calling"}) == [
        "RAG",
        "Milvus",
        "tool-calling",
    ]
    assert normalize_skill_area_from_text("普通开放问题") == ["agent"]


def test_build_skill_area_audit_sorts_keys() -> None:
    assert build_skill_area_audit(
        [
            {"question": "RAG 检索"},
            {"question": "Milvus 向量数据库"},
            {"question": "TypeScript Node.js"},
        ]
    ) == {"milvus": 1, "rag": 2, "typescript": 1}
