from app.domain.question_retriever import (
    extract_jd_skill_area,
    hybrid_rerank_questions,
    query_questions,
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


def test_query_questions_reranks_with_bm25_then_samples_from_top_candidates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.domain.question_retriever.random.sample",
        lambda population, sample_size: population[:sample_size],
    )

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
