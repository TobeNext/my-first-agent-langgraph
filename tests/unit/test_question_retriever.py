from app.domain.question_query import RetrievalQueryIntent
from app.domain.question_retriever import (
    extract_jd_skill_area,
    hybrid_rerank_questions,
    metadata_rerank_questions,
    query_questions,
    query_questions_multi,
)
from app.integrations.embeddings import (
    EMBEDDING_DIMENSION,
    HashEmbeddingProvider,
    build_embedding_provider,
    embed_query_text,
)
from app.schemas.interview_state import InterviewQuestionCandidate


class FakeStore:
    def search(self, *, vector, top_k, round_type):
        assert len(vector) == EMBEDDING_DIMENSION
        assert round_type == "professional-skills"
        return type(
            "Result",
            (),
            {
                "questions": [
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "q1",
                            "text": "请解释 Redis 队列的失败恢复。",
                            "score": 0.7,
                            "role": "professional-skills",
                            "skillArea": ["Redis"],
                        }
                    ),
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "q2",
                            "text": "请解释 RAG 检索链路。",
                            "score": 0.65,
                            "role": "professional-skills",
                            "skillArea": ["RAG"],
                        }
                    ),
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "q3",
                            "text": "请解释普通后端接口。",
                            "score": 0.6,
                            "role": "professional-skills",
                            "skillArea": ["java"],
                        }
                    ),
                ]
            },
        )()


def test_embedding_is_384_dimensional() -> None:
    vector = embed_query_text("RAG retrieval and Redis queue")

    assert len(vector) == 384
    assert any(value != 0 for value in vector)


def test_embedding_falls_back_to_hash_when_provider_has_no_key(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    monkeypatch.setenv("MODEL_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ZHIPU_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        provider = build_embedding_provider()
    finally:
        get_settings.cache_clear()

    assert isinstance(provider, HashEmbeddingProvider)


def test_query_questions_reranks_with_bm25_then_selects_top_candidates() -> None:
    result = query_questions(
        query_text="Target role: AI Engineer\nPrimary skill: RAG 检索",
        top_k=1,
        round_type="professional-skills",
        skill="RAG",
        store=FakeStore(),
    )

    assert result.count == 1
    assert [question.id for question in result.bm25Candidates] == ["q2", "q1", "q3"]
    assert result.questions[0].id == "q2"


def test_extract_jd_skill_area_drops_default_agent() -> None:
    assert extract_jd_skill_area("普通开放问题") == []
    assert extract_jd_skill_area("RAG 检索与 Milvus 向量数据库") == ["rag", "milvus"]


def test_hybrid_rerank_keeps_bm25_compatibility_as_skill_area_score() -> None:
    candidates = [
        InterviewQuestionCandidate.model_validate(
            {
                "id": "high-vector-no-skill",
                "text": "请解释 Redis 队列。",
                "score": 0.9,
                "role": "professional-skills",
                "skillArea": ["Redis"],
            }
        ),
        InterviewQuestionCandidate.model_validate(
            {
                "id": "lower-vector-rag",
                "text": "请解释 RAG 检索链路。",
                "score": 0.85,
                "role": "professional-skills",
                "skillArea": ["rag"],
            }
        ),
        InterviewQuestionCandidate.model_validate(
            {
                "id": "lowest-vector-no-skill",
                "text": "请解释普通后端接口。",
                "score": 0.8,
                "role": "professional-skills",
                "skillArea": ["java"],
            }
        ),
    ]

    reranked = hybrid_rerank_questions(
        candidates,
        query_text="Target role: AI Engineer\nPrimary skill: RAG 检索",
    )

    assert [question.id for question in reranked] == [
        "lower-vector-rag",
        "high-vector-no-skill",
        "lowest-vector-no-skill",
    ]


class MultiQueryStore:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, *, vector, top_k, round_type):
        self.calls += 1
        shared = InterviewQuestionCandidate.model_validate(
            {
                "id": "shared-rag-design",
                "text": "请设计一个 RAG Agent 系统，并说明工具调用和记忆管理。",
                "score": 0.7,
                "role": "professional-skills",
                "skillArea": ["rag", "tool-calling", "memory"],
                "questionType": "system_design",
                "difficulty": "hard",
                "tags": "RAG, Agent, Tool Calling, Memory",
            }
        )
        one_off = InterviewQuestionCandidate.model_validate(
            {
                "id": f"one-off-{self.calls}",
                "text": "请解释普通后端接口。",
                "score": 0.95,
                "role": "professional-skills",
                "skillArea": ["java"],
                "questionType": "knowledge-check",
                "difficulty": "medium",
            }
        )
        return type("Result", (), {"questions": [shared, one_off]})()


def test_query_questions_multi_uses_three_way_rrf_and_metadata_rerank() -> None:
    result = query_questions_multi(
        query_intents=[
            RetrievalQueryIntent("skill_exact", "Primary skill: RAG 检索 Tool Calling Memory"),
            RetrievalQueryIntent(
                "job_scenario",
                "Job responsibility signals: Agent 架构 工具调用 记忆管理",
            ),
            RetrievalQueryIntent(
                "capability_probe",
                "Capability focus: system design scenario hard",
            ),
        ],
        top_k=1,
        round_type="professional-skills",
        skill="RAG",
        store=MultiQueryStore(),
    )

    assert result.count == 1
    assert result.questions[0].id == "shared-rag-design"
    assert result.bm25Candidates[0].id == "shared-rag-design"


def test_metadata_rerank_prefers_matching_system_design_question() -> None:
    candidates = [
        InterviewQuestionCandidate.model_validate(
            {
                "id": "knowledge",
                "text": "请解释 RAG 的定义。",
                "score": 0.95,
                "role": "professional-skills",
                "skillArea": ["rag"],
                "questionType": "knowledge-check",
                "difficulty": "medium",
            }
        ),
        InterviewQuestionCandidate.model_validate(
            {
                "id": "design",
                "text": "请设计一个支持 Tool Calling 和 Memory 的 RAG Agent。",
                "score": 0.8,
                "role": "professional-skills",
                "skillArea": ["rag", "tool-calling", "memory"],
                "questionType": "system_design",
                "difficulty": "hard",
                "tags": "RAG, Tool Calling, Memory",
            }
        ),
    ]

    reranked = metadata_rerank_questions(
        candidates,
        query_text=(
            "Primary skill: RAG Tool Calling Memory\n"
            "Job responsibility signals: Agent 架构\n"
            "Expected difficulty: hard scenario"
        ),
    )

    assert [question.id for question in reranked] == ["design", "knowledge"]
