from app.integrations.milvus_store import MilvusQuestionStore, _normalize_milvus_uri


def test_normalize_milvus_uri_adds_http_scheme() -> None:
    assert _normalize_milvus_uri("localhost:19530") == "http://localhost:19530"
    assert _normalize_milvus_uri("http://localhost:19530") == "http://localhost:19530"
    assert _normalize_milvus_uri("tcp://localhost:19530") == "tcp://localhost:19530"


def test_search_reads_legacy_metadata_only_collection(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, uri: str, pool_size: int = 4):
            captured["uri"] = uri

        def describe_collection(self, collection_name: str) -> dict:
            captured["described"] = collection_name
            return {
                "fields": [
                    {"name": "id"},
                    {"name": "vector"},
                    {"name": "metadata"},
                ]
            }

        def search(self, **kwargs) -> list[list[dict]]:
            captured["search"] = kwargs
            return [
                [
                    {
                        "id": "q-rag",
                        "distance": 0.88,
                        "entity": {
                            "id": "q-rag",
                            "metadata": {
                                "question": "请解释 RAG 检索链路。",
                                "answer": "说明 query rewrite、召回、融合和重排。",
                                "text": (
                                    "# 请解释 RAG 检索链路。\n\n"
                                    "说明 query rewrite、召回、融合和重排。"
                                ),
                                "role": "AI Agent Engineer",
                                "difficulty": "medium",
                                "skillArea": ["RAG"],
                            },
                        },
                    },
                    {
                        "id": "q-project",
                        "distance": 0.86,
                        "entity": {
                            "id": "q-project",
                            "metadata": {
                                "question": "请介绍项目背景。",
                                "answer": "介绍背景、职责和结果。",
                                "text": "# 请介绍项目背景。",
                                "role": "project-experience",
                            },
                        },
                    },
                ]
            ]

    monkeypatch.setattr("app.integrations.milvus_store.MilvusClient", FakeClient)

    result = MilvusQuestionStore(address="localhost:19530").search(
        vector=[0.1, 0.2, 0.3],
        top_k=2,
        round_type="professional-skills",
    )

    assert captured["uri"] == "http://localhost:19530"
    assert captured["search"] == {
        "collection_name": "interview_questions",
        "data": [[0.1, 0.2, 0.3]],
        "limit": 2,
        "anns_field": "vector",
        "output_fields": ["id", "metadata"],
        "search_params": {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        },
        "filter": None,
    }
    assert [question.id for question in result.questions] == ["q-rag", "q-project"]
    assert result.questions[0].role == "AI Agent Engineer"
    assert result.questions[0].skillArea == ["RAG"]


def test_search_uses_global_metadata_filter_when_collection_supports_it(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, uri: str, pool_size: int = 4):
            captured["uri"] = uri

        def describe_collection(self, collection_name: str) -> dict:
            return {
                "fields": [
                    {"name": "id"},
                    {"name": "vector"},
                    {"name": "metadata"},
                    {"name": "role"},
                    {"name": "difficulty"},
                    {"name": "skillArea"},
                    {"name": "language"},
                    {"name": "isActive"},
                    {"name": "userId"},
                ]
            }

        def search(self, **kwargs) -> list[list[dict]]:
            captured["search"] = kwargs
            return [
                [
                    {
                        "id": "q-rag",
                        "score": 0.91,
                        "entity": {
                            "id": "q-rag",
                            "role": "professional-skills",
                            "difficulty": "medium",
                            "skillArea": ["RAG"],
                            "language": "zh",
                            "isActive": True,
                            "userId": "global",
                            "metadata": {
                                "question": "请解释 RAG 检索链路。",
                                "answer": "说明 query rewrite、召回、融合和重排。",
                                "text": "# 请解释 RAG 检索链路。",
                            },
                        },
                    }
                ]
            ]

    monkeypatch.setattr("app.integrations.milvus_store.MilvusClient", FakeClient)

    result = MilvusQuestionStore(address="http://localhost:19530").search(
        vector=[0.1, 0.2, 0.3],
        top_k=1,
        round_type="professional-skills",
    )

    assert captured["search"] == {
        "collection_name": "interview_questions",
        "data": [[0.1, 0.2, 0.3]],
        "limit": 1,
        "anns_field": "vector",
        "output_fields": [
            "id",
            "metadata",
            "role",
            "difficulty",
            "skillArea",
            "language",
            "isActive",
            "userId",
        ],
        "search_params": {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        },
        "filter": 'isActive == true and language == "zh" and userId == "global"',
    }
    assert [question.id for question in result.questions] == ["q-rag"]


def test_search_maps_stable_metadata_fields_to_candidate(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *, uri: str, pool_size: int = 4):
            pass

        def describe_collection(self, collection_name: str) -> dict:
            return {
                "fields": [
                    {"name": "id"},
                    {"name": "vector"},
                    {"name": "metadata"},
                    {"name": "role"},
                    {"name": "difficulty"},
                    {"name": "skillArea"},
                ]
            }

        def search(self, **kwargs) -> list[list[dict]]:
            return [
                [
                    {
                        "id": "q-agent",
                        "score": 0.93,
                        "entity": {
                            "id": "q-agent",
                            "role": "professional-skills",
                            "metadata": {
                                "question": "请设计 Agent 工具调用链路。",
                                "answer": "- 工具 schema\n- 错误处理",
                                "text": "# 请设计 Agent 工具调用链路。",
                                "answer_points": ["工具 schema", "错误处理"],
                                "tags": ["Agent", "Tool Calling"],
                                "skills": ["tool-calling", "workflow"],
                                "level": "senior",
                                "question_type": "system_design",
                                "job_family": "llm_agent_engineer",
                                "job_duties": ["工具调用", "自主执行"],
                                "language": "zh",
                                "embedding_text": "Agent 工具调用 自主执行",
                                "source": "seed-bank",
                            },
                        },
                    }
                ]
            ]

    monkeypatch.setattr("app.integrations.milvus_store.MilvusClient", FakeClient)

    result = MilvusQuestionStore(address="http://localhost:19530").search(
        vector=[0.1, 0.2, 0.3],
        top_k=1,
        round_type="professional-skills",
    )

    question = result.questions[0]
    assert question.id == "q-agent"
    assert question.questionType == "system_design"
    assert question.difficulty == 8
    assert question.skillArea == ["tool-calling", "workflow"]
    assert question.answerPoints == ["工具 schema", "错误处理"]
    assert question.skills == ["tool-calling", "workflow"]
    assert question.level == "senior"
    assert question.jobFamily == "llm_agent_engineer"
    assert question.jobDuties == ["工具调用", "自主执行"]
    assert question.language == "zh"
    assert question.embeddingText == "Agent 工具调用 自主执行"
    assert question.source == "seed-bank"
    assert question.isActive is True
    assert question.userId == "global"


def test_search_filters_bad_or_inactive_metadata_in_application(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *, uri: str, pool_size: int = 4):
            pass

        def describe_collection(self, collection_name: str) -> dict:
            return {
                "fields": [
                    {"name": "id"},
                    {"name": "vector"},
                    {"name": "metadata"},
                ]
            }

        def search(self, **kwargs) -> list[list[dict]]:
            return [
                [
                    {
                        "id": "good",
                        "score": 0.93,
                        "entity": {
                            "metadata": {
                                "question": "请解释 RAG 检索链路。",
                                "answer": "说明 query rewrite、召回、融合和重排。",
                                "text": "# 请解释 RAG 检索链路。",
                            },
                        },
                    },
                    {
                        "id": "missing-answer",
                        "score": 0.92,
                        "entity": {
                            "metadata": {
                                "question": "请解释 BM25。",
                                "text": "# 请解释 BM25。",
                            },
                        },
                    },
                    {
                        "id": "inactive",
                        "score": 0.91,
                        "entity": {
                            "metadata": {
                                "question": "请解释 Memory。",
                                "answer": "说明记忆机制。",
                                "text": "# 请解释 Memory。",
                                "isActive": False,
                            },
                        },
                    },
                    {
                        "id": "english",
                        "score": 0.9,
                        "entity": {
                            "metadata": {
                                "question": "Explain Tool Calling.",
                                "answer": "Schema and error handling.",
                                "text": "# Explain Tool Calling.",
                                "language": "en",
                            },
                        },
                    },
                    {
                        "id": "private",
                        "score": 0.89,
                        "entity": {
                            "metadata": {
                                "question": "请解释私有题库。",
                                "answer": "说明私有题库。",
                                "text": "# 请解释私有题库。",
                                "userId": "user-1",
                            },
                        },
                    },
                ]
            ]

    monkeypatch.setattr("app.integrations.milvus_store.MilvusClient", FakeClient)

    result = MilvusQuestionStore(address="http://localhost:19530").search(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        round_type="professional-skills",
    )

    assert [question.id for question in result.questions] == ["good"]
    assert result.questions[0].language == "zh"
    assert result.questions[0].isActive is True
    assert result.questions[0].userId == "global"
