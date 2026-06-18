from app.integrations.milvus_store import MilvusQuestionStore, _normalize_milvus_uri


def test_normalize_milvus_uri_adds_http_scheme() -> None:
    assert _normalize_milvus_uri("localhost:19530") == "http://localhost:19530"
    assert _normalize_milvus_uri("http://localhost:19530") == "http://localhost:19530"
    assert _normalize_milvus_uri("tcp://localhost:19530") == "tcp://localhost:19530"


def test_search_reads_legacy_metadata_only_collection(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, uri: str):
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
                                "role": "project-experience",
                            },
                        },
                    },
                ]
            ]

    monkeypatch.setattr("pymilvus.MilvusClient", FakeClient)

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
        "filter": None,
    }
    assert [question.id for question in result.questions] == ["q-rag"]
    assert result.questions[0].role == "professional-skills"
    assert result.questions[0].skillArea == ["RAG"]


def test_search_uses_scalar_role_filter_when_collection_supports_it(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, uri: str):
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
                            "metadata": {"question": "请解释 RAG 检索链路。"},
                        },
                    }
                ]
            ]

    monkeypatch.setattr("pymilvus.MilvusClient", FakeClient)

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
        "output_fields": ["id", "metadata", "role", "difficulty", "skillArea"],
        "filter": 'role == "professional-skills"',
    }
    assert [question.id for question in result.questions] == ["q-rag"]
