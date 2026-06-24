from app.domain.question_metadata import (
    build_skill_area_audit,
    clean_interview_question_metadata,
    normalize_difficulty_score,
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

    assert cleaned["questionType"] == "knowledge_check"
    assert cleaned["source"] == "interview-question-bank"
    assert cleaned["role"] == "professional-skills"
    assert cleaned["difficulty"] == 8
    assert cleaned["level"] == "senior"
    assert cleaned["question_type"] == "knowledge_check"
    assert cleaned["answer_points"] == ["Use vector search and reranking."]
    assert cleaned["tags"] == ["RAG", "Milvus"]
    assert cleaned["skills"] == ["rag", "milvus", "bm25"]
    assert cleaned["skillArea"] == ["rag", "milvus", "bm25"]
    assert cleaned["language"] == "zh"
    assert cleaned["isActive"] is True
    assert cleaned["userId"] == "global"
    assert "mainCategory" not in cleaned
    assert "subCategory" not in cleaned
    assert "company" not in cleaned
    assert "mainCategory" not in cleaned["metadata"]
    assert "subCategory" not in cleaned["metadata"]
    assert "company" not in cleaned["metadata"]
    assert "tags" not in cleaned["metadata"]
    assert "role" not in cleaned["metadata"]


def test_clean_metadata_supports_stable_question_bank_fields() -> None:
    cleaned = clean_interview_question_metadata(
        {
            "question": "请设计 Agent 工具调用链路。",
            "answer": "- 工具 schema\n- 错误处理",
            "answer_points": ["工具 schema", "错误处理"],
            "tags": "Agent, Tool Calling",
            "skills": ["tool-calling", "workflow"],
            "level": "senior",
            "question_type": "system_design",
            "job_family": "llm_agent_engineer",
            "job_duties": ["工具调用", "自主执行"],
            "language": "zh",
            "embedding_text": "Agent 工具调用 自主执行",
            "source": "seed-bank",
        }
    )

    assert cleaned["questionType"] == "system_design"
    assert cleaned["difficulty"] == 8
    assert cleaned["skillArea"] == ["tool-calling", "workflow"]
    assert cleaned["skills"] == ["tool-calling", "workflow"]
    assert cleaned["answer_points"] == ["工具 schema", "错误处理"]
    assert cleaned["job_family"] == "llm_agent_engineer"
    assert cleaned["job_duties"] == ["工具调用", "自主执行"]
    assert cleaned["embedding_text"] == "Agent 工具调用 自主执行"
    assert cleaned["source"] == "seed-bank"
    assert cleaned["language"] == "zh"
    assert cleaned["isActive"] is True
    assert cleaned["userId"] == "global"
    assert cleaned["metadata"] == {}


def test_clean_metadata_preserves_explicit_global_filter_fields() -> None:
    cleaned = clean_interview_question_metadata(
        {
            "question": "请解释上下文和记忆的区别。",
            "answer": "上下文是当前输入，记忆是可复用信息资产。",
            "text": "# 请解释上下文和记忆的区别。",
            "questionType": "system-design",
            "difficulty": 10,
            "language": "en",
            "isActive": "false",
            "userId": "global",
        }
    )

    assert cleaned["questionType"] == "system_design"
    assert cleaned["difficulty"] == 10
    assert cleaned["language"] == "en"
    assert cleaned["isActive"] is False
    assert cleaned["userId"] == "global"


def test_normalize_difficulty_score_maps_legacy_values() -> None:
    assert normalize_difficulty_score("easy") == 3
    assert normalize_difficulty_score("medium") == 6
    assert normalize_difficulty_score("hard") == 8
    assert normalize_difficulty_score("unknown") == 5
    assert normalize_difficulty_score(12) == 10
    assert normalize_difficulty_score(0) == 1


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
