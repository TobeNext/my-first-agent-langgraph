import app.domain.question_retriever as question_retriever
from app.domain.question_planner import ProfessionalQuestionPlan
from app.domain.question_query import RetrievalQueryIntent
from app.domain.question_retriever import (
    _weighted_sample_questions,
    extract_jd_skill_area,
    hybrid_rerank_questions,
    metadata_rerank_questions,
    query_questions,
    query_questions_multi,
    retrieve_initialization_questions,
)
from app.domain.resume_jd_match import ResumeJdMatchAnalysis
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


def test_query_questions_reranks_with_bm25_then_selects_top_candidates(monkeypatch) -> None:
    monkeypatch.setattr(question_retriever.random, "random", lambda: 0.0)

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


class EmptyVectorStore:
    def search(self, *, vector, top_k, round_type):
        return type("Result", (), {"questions": []})()


class KeywordOnlyStore:
    def search(self, *, query_text, top_k, round_type):
        assert "Tool Calling" in query_text
        assert round_type == "professional-skills"
        return type(
            "Result",
            (),
            {
                "questions": [
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "keyword-tool-calling",
                            "text": "请设计 Agent Tool Calling 的错误恢复链路。",
                            "score": 0.4,
                            "role": "professional-skills",
                            "skillArea": ["tool-calling"],
                            "questionType": "system_design",
                            "difficulty": "hard",
                            "tags": "Tool Calling, Agent",
                        }
                    )
                ]
            },
        )()


def test_query_questions_multi_merges_keyword_only_hits_with_rrf() -> None:
    result = query_questions_multi(
        query_intents=[
            RetrievalQueryIntent(
                "job_scenario",
                "Job responsibility signals: Agent Tool Calling 错误恢复",
            )
        ],
        top_k=1,
        round_type="professional-skills",
        skill="Tool Calling",
        store=EmptyVectorStore(),
        keyword_store=KeywordOnlyStore(),
    )

    assert result.count == 1
    assert result.questions[0].id == "keyword-tool-calling"
    assert result.bm25Candidates[0].id == "keyword-tool-calling"


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


def test_metadata_rerank_uses_rrf_and_question_type_without_difficulty() -> None:
    candidates = [
        InterviewQuestionCandidate.model_validate(
            {
                "id": "culture-hard",
                "text": "请描述一次团队协作经历。",
                "score": 0.9,
                "role": "professional-skills",
                "questionType": "culture_fit",
                "difficulty": 10,
            }
        ),
        InterviewQuestionCandidate.model_validate(
            {
                "id": "design-easy",
                "text": "请设计一个 Agent Memory 机制。",
                "score": 0.8,
                "role": "professional-skills",
                "questionType": "system_design",
                "difficulty": 1,
            }
        ),
    ]

    reranked = metadata_rerank_questions(
        candidates,
        query_text="Agent Memory system design",
        rrf_scores={"culture-hard": 1.0, "design-easy": 1.0},
    )

    assert [question.id for question in reranked] == ["design-easy", "culture-hard"]


class DuplicateVetoStore:
    def search(self, *, vector, top_k, round_type):
        return type(
            "Result",
            (),
            {
                "questions": [
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "memory-primary",
                            "text": "Claude Code 的记忆架构是什么？上下文是否等于记忆？",
                            "score": 0.95,
                            "role": "professional-skills",
                            "questionType": "system_design",
                        }
                    ),
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "memory-near-duplicate",
                            "text": "Claude Code 的记忆架构是什么？上下文是否等于记忆？",
                            "score": 0.94,
                            "role": "professional-skills",
                            "questionType": "system_design",
                        }
                    ),
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "rag-next",
                            "text": "请解释 RAG 检索链路的 query rewrite 和 RRF 融合。",
                            "score": 0.7,
                            "role": "professional-skills",
                            "questionType": "technical",
                        }
                    ),
                ]
            },
        )()


def test_query_questions_vetoes_duplicate_candidates_before_top_k() -> None:
    result = query_questions(
        query_text="Claude Code 记忆架构 上下文 记忆",
        top_k=2,
        round_type="professional-skills",
        skill="memory",
        store=DuplicateVetoStore(),
    )

    assert [question.id for question in result.questions] == ["memory-primary", "rag-next"]
    vetoed = [
        question
        for question in result.bm25Candidates
        if question.id == "memory-near-duplicate"
    ]
    assert vetoed


def test_metadata_rerank_marks_duplicate_veto_candidates_for_trace() -> None:
    candidates = DuplicateVetoStore().search(
        vector=[0.1],
        top_k=3,
        round_type="professional-skills",
    ).questions

    trace = retrieve_initialization_questions(
        selected_direction="AI Agent 工程师",
        raw_kickoff_message="",
        professional_skills="- Claude Code Memory",
        normalized_professional_skills=["Memory"],
        project_experience="",
        normalized_project_topics=[],
        job_description="",
        professional_question_plan=[_plan("Memory"), _plan("RAG")],
        store=DuplicateVetoStore(),
    )

    assert candidates[1].id == "memory-near-duplicate"
    duplicate_candidates = [
        candidate
        for candidate in trace.recallTraces[0].candidates
        if candidate["id"] == "memory-near-duplicate"
    ]
    assert duplicate_candidates
    assert duplicate_candidates[0]["isDuplicate"] is True
    assert duplicate_candidates[0]["filterReason"] == "duplicate-veto"


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def test_weighted_sample_uses_selection_score() -> None:
    candidates = [
        InterviewQuestionCandidate.model_validate(
            {
                "id": "low",
                "text": "低分候选",
                "selectionScore": 0.1,
            }
        ),
        InterviewQuestionCandidate.model_validate(
            {
                "id": "high",
                "text": "高分候选",
                "selectionScore": 0.9,
            }
        ),
    ]

    selected = _weighted_sample_questions(
        candidates,
        1,
        random_source=FixedRandom(0.99),
    )

    assert [question.id for question in selected] == ["high"]


def test_weighted_sample_only_draws_from_top_five() -> None:
    candidates = [
        InterviewQuestionCandidate.model_validate(
            {
                "id": f"candidate-{index}",
                "text": f"候选 {index}",
                "selectionScore": 0.1,
            }
        )
        for index in range(1, 6)
    ]
    candidates.append(
        InterviewQuestionCandidate.model_validate(
            {
                "id": "outside-top-five",
                "text": "第六个候选",
                "selectionScore": 100.0,
            }
        )
    )

    selected = _weighted_sample_questions(
        candidates,
        1,
        random_source=FixedRandom(0.99),
    )

    assert selected[0].id != "outside-top-five"


def test_weighted_sample_handles_zero_scores() -> None:
    candidates = [
        InterviewQuestionCandidate.model_validate(
            {
                "id": "zero",
                "text": "零分候选",
                "selectionScore": 0.0,
            }
        )
    ]

    selected = _weighted_sample_questions(candidates, 1, random_source=FixedRandom(0.5))

    assert [question.id for question in selected] == ["zero"]


def test_recall_trace_includes_metadata_score_breakdown() -> None:
    result = query_questions_multi(
        query_intents=[
            RetrievalQueryIntent(
                "job_scenario",
                "Job responsibility signals: Agent 工具调用 自主执行 hard scenario",
            )
        ],
        top_k=1,
        round_type="professional-skills",
        skill="Tool Calling",
        store=TraceStore(),
    )
    trace_result = retrieve_initialization_questions(
        selected_direction="AI Agent 工程师",
        raw_kickoff_message="",
        professional_skills="- Tool Calling",
        normalized_professional_skills=["Tool Calling"],
        project_experience="",
        normalized_project_topics=[],
        job_description="",
        professional_question_plan=[_plan("Tool Calling")],
        store=TraceStore(),
    )

    assert result.questions[0].id == "trace-tool"
    candidate = trace_result.recallTraces[0].candidates[0]
    assert set(candidate["scoreBreakdown"]) == {"rrf", "questionType"}
    assert candidate["scoreBreakdown"]["rrf"] == 1.0
    assert candidate["scoreBreakdown"]["questionType"] == 1.0
    assert candidate["matchedMetadata"]["questionType"] == "system_design"
    assert candidate["matchedMetadata"]["level"] == "senior"
    assert candidate["isDuplicate"] is False


class TraceStore:
    def search(self, *, vector, top_k, round_type):
        return type(
            "Result",
            (),
            {
                "questions": [
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": "trace-tool",
                            "text": "请设计 Agent Tool Calling 的错误恢复链路。",
                            "score": 0.9,
                            "role": "professional-skills",
                            "skillArea": ["tool-calling"],
                            "skills": ["tool-calling"],
                            "questionType": "system_design",
                            "difficulty": "hard",
                            "level": "senior",
                            "jobDuties": ["工具调用", "自主执行"],
                            "tags": "Tool Calling, Agent",
                        }
                    )
                ]
            },
        )()


class SegmentStore:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, *, vector, top_k, round_type):
        self.calls += 1
        if self.calls <= 3:
            question_id = "match-question"
            text = "请说明你如何把 RAG 检索能力用于 JD 中的 Agent 场景。"
        elif self.calls == 4:
            question_id = "resume-only-question"
            text = "请说明你简历中的 Vue 项目经验。"
        elif self.calls == 5:
            question_id = "jd-only-question"
            text = "请说明你如何补齐 JD 中的模型评估能力。"
        else:
            question_id = "project-question"
            text = "请说明你的项目经验。"
        return type(
            "Result",
            (),
            {
                "questions": [
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": question_id,
                            "text": text,
                            "score": 0.9,
                            "role": round_type,
                            "skillArea": ["rag"],
                        }
                    )
                ]
            },
        )()


def test_retrieve_initialization_questions_uses_three_match_sections() -> None:
    store = SegmentStore()
    analysis = ResumeJdMatchAnalysis.model_validate(
        {
            "resumeJdMatch": [
                {
                    "resumeSignal": "RAG 检索",
                    "jobSignal": "Agent 检索增强",
                    "matchType": "skill",
                    "relevance": 0.9,
                    "priority": "high",
                    "evidence": {
                        "resumeSignals": ["RAG 检索"],
                        "jobSignals": ["Agent 检索增强"],
                        "projectSignals": [],
                    },
                    "interviewFocus": ["RAG 检索"],
                    "suggestedQuestionTypes": ["experience_probe"],
                }
            ],
            "resumeOnly": [
                {"resumeSignal": "Vue", "category": "skill", "evidence": ["Vue"]}
            ],
            "jdOnly": [
                {
                    "jobSignal": "模型评估",
                    "category": "requirement",
                    "priority": "medium",
                    "evidence": ["模型评估"],
                }
            ],
            "isJobMatched": True,
            "mismatchReason": None,
        }
    )

    result = retrieve_initialization_questions(
        selected_direction="AI Agent 工程师",
        raw_kickoff_message="",
        professional_skills="- RAG 检索\n- Vue",
        normalized_professional_skills=["RAG 检索", "Vue"],
        project_experience="",
        normalized_project_topics=[],
        job_description="- Agent 检索增强\n- 模型评估",
        professional_question_plan=[_plan("RAG"), _plan("Vue"), _plan("评估"), _plan("Agent")],
        match_analysis=analysis,
        store=store,
    )

    assert [question.id for question in result.professionalQuestions] == [
        "match-question",
        "resume-only-question",
        "jd-only-question",
    ]
    assert [trace.skill.split(":", 1)[0] for trace in result.recallTraces[:3]] == [
        "resume-jd-match",
        "resume-only",
        "jd-only",
    ]


class DuplicateSegmentStore:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, *, vector, top_k, round_type):
        self.calls += 1
        question_id = "shared-question"
        text = "请说明你如何把 RAG 检索能力用于 JD 中的 Agent 场景。"
        return type(
            "Result",
            (),
            {
                "questions": [
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": question_id,
                            "text": text,
                            "score": 0.9,
                            "role": round_type,
                            "skillArea": ["rag"],
                        }
                    )
                ]
            },
        )()


def test_retrieve_initialization_questions_filters_duplicate_segment_questions() -> None:
    analysis = ResumeJdMatchAnalysis.model_validate(
        {
            "resumeJdMatch": [
                {
                    "resumeSignal": "RAG 检索",
                    "jobSignal": "Agent 检索增强",
                    "matchType": "skill",
                    "relevance": 0.9,
                    "priority": "high",
                    "evidence": {
                        "resumeSignals": ["RAG 检索"],
                        "jobSignals": ["Agent 检索增强"],
                        "projectSignals": [],
                    },
                    "interviewFocus": ["RAG 检索"],
                    "suggestedQuestionTypes": ["experience_probe"],
                }
            ],
            "resumeOnly": [
                {"resumeSignal": "RAG 项目", "category": "project", "evidence": ["RAG 项目"]}
            ],
            "isJobMatched": True,
            "mismatchReason": None,
        }
    )

    result = retrieve_initialization_questions(
        selected_direction="AI Agent 工程师",
        raw_kickoff_message="",
        professional_skills="- RAG 检索",
        normalized_professional_skills=["RAG 检索"],
        project_experience="",
        normalized_project_topics=[],
        job_description="- Agent 检索增强",
        professional_question_plan=[_plan("RAG"), _plan("RAG 项目")],
        match_analysis=analysis,
        store=DuplicateSegmentStore(),
    )

    assert [question.id for question in result.professionalQuestions] == ["shared-question"]
    assert result.recallTraces[1].selectedQuestionIds == []
    assert result.recallTraces[1].candidates[0]["isDuplicate"] is True
    assert set(result.recallTraces[1].candidates[0]["scoreBreakdown"]) == {
        "rrf",
        "questionType",
    }


def _plan(skill: str) -> ProfessionalQuestionPlan:
    return ProfessionalQuestionPlan(
        kind="skill-focus",
        primarySkill=skill,
        relatedSkills=[],
        lens="implementation-depth",
        targetAbility=skill,
        questionType="knowledge-check",
        coverageIntent="implementation-depth",
        resumeSignals=[skill],
        jobDescriptionSignals=[],
        questionDriver="resume",
        expectedDifficulty="medium",
        selectionReason="test plan",
    )
