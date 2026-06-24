import json

from app.config import Settings
from app.domain.resume_jd_match import (
    ResumeJdMatchAnalysis,
    build_resume_jd_match_analysis,
)
from app.integrations.models import TracedChatModel


class StructuredMatchModel:
    def with_structured_output(self, schema):
        class _Structured:
            def invoke(self, prompt):
                return schema.model_validate(
                    {
                        "resumeJdMatch": [
                            {
                                "resumeSignal": "RAG 检索",
                                "jobSignal": "负责 RAG 检索增强生成",
                                "matchType": "skill",
                                "relevance": 0.9,
                                "priority": "high",
                                "evidence": {
                                    "resumeSignals": ["RAG 检索"],
                                    "jobSignals": ["负责 RAG 检索增强生成"],
                                    "projectSignals": ["AI 面试 Agent"],
                                },
                                "interviewFocus": ["RAG 检索"],
                                "suggestedQuestionTypes": ["experience_probe"],
                            }
                        ],
                        "resumeOnly": [
                            {
                                "resumeSignal": "Vue",
                                "category": "skill",
                                "evidence": ["Vue"],
                            }
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

        return _Structured()


class RawJsonModel:
    def invoke(self, prompt, **kwargs):
        return json.dumps(
            {
                "resumeJdMatch": [],
                "resumeOnly": [{"resumeSignal": "Vue", "category": "skill", "evidence": []}],
                "jdOnly": [
                    {
                        "jobSignal": "Java 后端",
                        "category": "requirement",
                        "priority": "high",
                        "evidence": ["Java 后端"],
                    }
                ],
                "isJobMatched": False,
                "mismatchReason": "岗位不匹配",
            },
            ensure_ascii=False,
        )


class DeepSeekRawJsonModel(RawJsonModel):
    def __init__(self) -> None:
        self.invoke_kwargs: dict[str, object] | None = None

    def invoke(self, prompt, **kwargs):
        self.invoke_kwargs = kwargs
        return super().invoke(prompt, **kwargs)

    def with_structured_output(self, schema):
        raise AssertionError("DeepSeek should not use native structured output by default")


def _settings(**overrides: object) -> Settings:
    values = {
        "APP_ENV": "test",
        "MODEL_PROVIDER": "mock",
        "MODEL_NAME": "mock/interview-runtime",
        "MODEL_BASE_URL": None,
        "MODEL_STRUCTURED_OUTPUT_MODE": "auto",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_build_resume_jd_match_analysis_uses_structured_llm_output() -> None:
    analysis = build_resume_jd_match_analysis(
        professional_skills="- RAG 检索\n- Vue",
        project_experience="- AI 面试 Agent",
        job_description="- 负责 RAG 检索增强生成\n- 模型评估",
        normalized_skills=["RAG 检索", "Vue"],
        normalized_project_topics=["AI 面试 Agent"],
        chat_model=StructuredMatchModel(),
    )

    assert isinstance(analysis, ResumeJdMatchAnalysis)
    assert analysis.isJobMatched is True
    assert analysis.resumeJdMatch[0].resumeSignal == "RAG 检索"
    assert analysis.resumeOnly[0].resumeSignal == "Vue"
    assert analysis.jdOnly[0].jobSignal == "模型评估"


def test_build_resume_jd_match_analysis_uses_raw_json_for_deepseek_reasoner() -> None:
    inner_model = DeepSeekRawJsonModel()

    analysis = build_resume_jd_match_analysis(
        professional_skills="- Vue",
        project_experience="",
        job_description="- Java 后端",
        normalized_skills=["Vue"],
        normalized_project_topics=[],
        chat_model=TracedChatModel(
            model=inner_model,
            settings=_settings(
                MODEL_PROVIDER="deepseek",
                MODEL_NAME="deepseek-reasoner",
                MODEL_API_KEY="test-key",
            ),
        ),
    )

    assert analysis.resumeJdMatch == []
    assert analysis.isJobMatched is False
    assert analysis.mismatchReason == "岗位不匹配"
    assert inner_model.invoke_kwargs == {"response_format": {"type": "json_object"}}


def test_build_resume_jd_match_analysis_marks_empty_match_as_mismatch() -> None:
    analysis = build_resume_jd_match_analysis(
        professional_skills="- Vue",
        project_experience="",
        job_description="- Java 后端",
        normalized_skills=["Vue"],
        normalized_project_topics=[],
        chat_model=RawJsonModel(),
    )

    assert analysis.resumeJdMatch == []
    assert analysis.isJobMatched is False
    assert analysis.mismatchReason == "岗位不匹配"


def test_build_resume_jd_match_analysis_keeps_no_jd_as_resume_only() -> None:
    analysis = build_resume_jd_match_analysis(
        professional_skills="- RAG 检索",
        project_experience="- AI 面试 Agent",
        job_description="",
        normalized_skills=["RAG 检索"],
        normalized_project_topics=["AI 面试 Agent"],
    )

    assert analysis.isJobMatched is True
    assert analysis.resumeJdMatch == []
    assert [item.resumeSignal for item in analysis.resumeOnly] == ["RAG 检索", "AI 面试 Agent"]
