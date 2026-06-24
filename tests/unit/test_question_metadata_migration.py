from scripts.migrate_question_metadata import (
    build_milvus_scalar_fields,
    migrate_question_metadata,
    migrate_record,
    migrate_records,
)


def _legacy_metadata() -> dict:
    return {
        "question": "Claude Code 的记忆架构是什么？上下文是否等于记忆？",
        "answer": (
            "1. 上下文是当前模型输入，记忆是跨轮次、跨会话可复用的信息资产。"
        ),
        "questionType": "system-design",
        "company": "general",
        "role": "AI Agent Engineer",
        "difficulty": "hard",
        "source": "ai-agent-summary",
        "tags": [
            "三、Agent 核心机制：ReAct、CoT、Few-shot、Memory、Planning、Tool Use",
            "3.2 Memory 与 Prompt 设计",
            "memory",
        ],
        "mainCategory": "三、Agent 核心机制：ReAct、CoT、Few-shot、Memory、Planning、Tool Use",
        "subCategory": "3.2 Memory 与 Prompt 设计",
        "sourceFile": r"C:\Users\Blaine.Yu\Documents\Notes\AI Agent.md",
        "text": "# Claude Code 的记忆架构是什么？上下文是否等于记忆？",
    }


def test_migrate_question_metadata_removes_legacy_fields_and_adds_defaults() -> None:
    migrated = migrate_question_metadata(_legacy_metadata())

    assert migrated["questionType"] == "system_design"
    assert migrated["difficulty"] == 8
    assert migrated["language"] == "zh"
    assert migrated["isActive"] is True
    assert migrated["userId"] == "global"
    assert migrated["role"] == "AI Agent Engineer"
    assert migrated["source"] == "ai-agent-summary"
    assert migrated["tags"][-1] == "memory"
    assert "mainCategory" not in migrated
    assert "subCategory" not in migrated
    assert "company" not in migrated
    assert "mainCategory" not in migrated["metadata"]
    assert "subCategory" not in migrated["metadata"]
    assert "company" not in migrated["metadata"]


def test_build_milvus_scalar_fields_for_new_contract() -> None:
    scalar_fields = build_milvus_scalar_fields(_legacy_metadata())

    assert scalar_fields == {
        "difficulty": 8,
        "isActive": True,
        "language": "zh",
        "questionType": "system_design",
        "userId": "global",
    }


def test_migrate_record_updates_nested_metadata_and_top_level_scalars() -> None:
    migrated = migrate_record({"id": "q-memory", "metadata": _legacy_metadata()})

    assert migrated["id"] == "q-memory"
    assert migrated["metadata"]["questionType"] == "system_design"
    assert migrated["metadata"]["difficulty"] == 8
    assert migrated["difficulty"] == 8
    assert migrated["language"] == "zh"
    assert migrated["isActive"] is True
    assert migrated["userId"] == "global"


def test_migrate_records_accepts_plain_metadata_records() -> None:
    migrated = migrate_records([_legacy_metadata()])

    assert len(migrated) == 1
    assert migrated[0]["questionType"] == "system_design"
