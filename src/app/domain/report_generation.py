from __future__ import annotations

import json
import re
from typing import Any

from app.integrations.llm_logging import log_llm_error, log_llm_input, log_llm_output
from app.integrations.models import ChatModelLike, create_chat_model
from app.schemas.interview_report import ReportGenerationOutput

REPORT_GENERATION_PROMPT_VERSION = "report-generation-v1"


def build_report_generation_prompt(
    *,
    task: Any,
    interview_metadata: dict[str, Any],
    evaluation_results: list[dict[str, Any]],
    question_answer_context: list[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            _build_report_generation_system_prompt(),
            "Interview metadata:",
            json.dumps(interview_metadata, ensure_ascii=False, indent=2),
            "Evaluation results:",
            json.dumps(evaluation_results, ensure_ascii=False, indent=2),
            "Question and answer context:",
            json.dumps(question_answer_context, ensure_ascii=False, indent=2),
            "Report task:",
            json.dumps(_dump_generation_context(task), ensure_ascii=False, indent=2),
            (
                "Write a markdown interview report and structured per-answer review. "
                "For main-question answers, compare against evaluationPoints/referenceAnswer "
                "coverage. For follow-up answers, evaluate directness, technical_depth, "
                "evidence_specificity, and clarity_structure. Only include missing points "
                "when missingPoints is non-empty."
            ),
        ]
    )


async def generate_report_with_model(
    *,
    prompt: str,
    task: Any,
    model: ChatModelLike | None = None,
) -> ReportGenerationOutput:
    chat_model = model or create_chat_model()
    if _is_mock_chat_model(chat_model):
        return _build_mock_report_generation_output(task)
    if not hasattr(chat_model, "with_structured_output"):
        raise RuntimeError("Configured chat model does not support structured output.")

    thread_id = _context_field(task, "threadId")
    metadata = {
        "generationId": _generation_id(task),
        "interviewId": _context_field(task, "interviewId"),
        "responseLanguage": _context_field(task, "responseLanguage"),
        "promptVersion": REPORT_GENERATION_PROMPT_VERSION,
    }
    log_llm_input(
        thread_id=thread_id,
        operation="report-generation",
        prompt=prompt,
        metadata=metadata,
    )
    structured_model = chat_model.with_structured_output(ReportGenerationOutput)
    try:
        try:
            result = structured_model.invoke(prompt)
        except Exception as exc:
            log_llm_error(
                thread_id=thread_id,
                operation="report-generation",
                error=exc,
                metadata={**metadata, "stage": "structured-output"},
            )
            result = _parse_raw_model_json(chat_model.invoke(prompt))
        parsed = ReportGenerationOutput.model_validate(result)
        log_llm_output(
            thread_id=thread_id,
            operation="report-generation",
            output=parsed,
            metadata=metadata,
        )
        return parsed
    except Exception as exc:
        log_llm_error(
            thread_id=thread_id,
            operation="report-generation",
            error=exc,
            metadata=metadata,
        )
        raise


def _build_report_generation_system_prompt() -> str:
    return "\n".join(
        [
            "You are a senior technical interviewer writing a post-interview report.",
            "Use Chinese when responseLanguage is zh; otherwise use English.",
            "Return JSON only and follow the provided schema exactly.",
            "Do not reveal full reference answers or quote them as standard answers.",
            "Use referenceAnswer and evaluationPoints only to judge coverage.",
            "Be specific, fair, and actionable.",
            "",
            "For each candidate answer:",
            "- If targetType is main-question, compare the answer against the retrieved "
            "main question's evaluationPoints and referenceAnswer.",
            "- Identify missingPoints only for important expected points that were not covered.",
            "- If there are no missing points, return an empty missingPoints array and do not "
            "write a missing-points sentence in markdown.",
            "- If targetType is follow-up, grade it across these four aspects:",
            "  1. directness: whether it directly answers the follow-up question.",
            "  2. technical_depth: whether it explains mechanisms, trade-offs, edge cases, "
            "or constraints.",
            "  3. evidence_specificity: whether it uses concrete project evidence, "
            "implementation details, metrics, or examples.",
            "  4. clarity_structure: whether the answer is structured and easy to follow.",
            "- Use the existing answer-evaluation result as scoring evidence, but write your "
            "own concise interviewer comment.",
            "- Do not invent candidate experience that was not in the answer.",
            "- Do not include full reference answers in the report.",
            "",
            "Return exactly one JSON object that matches this schema:",
            "{",
            '  "summary": {',
            '    "overallScore": 8,',
            '    "overallComment": "...",',
            '    "strengths": ["..."],',
            '    "improvementPriorities": ["..."]',
            "  },",
            '  "questionReviews": [',
            "    {",
            '      "questionId": "node-id",',
            '      "attemptId": "attempt-id",',
            '      "targetType": "main-question",',
            '      "question": "...",',
            '      "score": 8,',
            '      "comment": "...",',
            '      "missingPoints": ["..."],',
            '      "improvementAdvice": ["..."]',
            "    }",
            "  ],",
            '  "markdown": "# 面试评估报告\\n\\n..."',
            "}",
            "Do not return a top-level report field.",
            "Do not return markdown without summary and questionReviews.",
            "questionReviews must cover every item in question_answer_context.",
            "Each questionReviews.attemptId must exactly match an input attemptId.",
            "targetType must be either main-question or follow-up.",
        ]
    )


def _parse_raw_model_json(value: Any) -> dict[str, Any]:
    content = _extract_model_content(value)
    if not content:
        raise ValueError("Model returned an empty response.")
    json_text = _extract_json_object_text(content)
    if not json_text:
        raise ValueError("Model response did not contain a JSON object.")
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def _extract_model_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts)
    return None


def _extract_json_object_text(text: str) -> str | None:
    trimmed = text.strip()
    if not trimmed:
        return None
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed, flags=re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else trimmed
    start_index = candidate.find("{")
    end_index = candidate.rfind("}")
    if start_index < 0 or end_index <= start_index:
        return None
    return candidate[start_index : end_index + 1]


def _is_mock_chat_model(model: Any) -> bool:
    return model.__class__.__name__ == "MockChatModel"


def _dump_generation_context(context: Any) -> dict[str, Any]:
    if hasattr(context, "model_dump"):
        return context.model_dump(exclude_none=True)
    if isinstance(context, dict):
        return {key: value for key, value in context.items() if value is not None}
    return {
        key: getattr(context, key)
        for key in (
            "generationId",
            "taskId",
            "interviewId",
            "threadId",
            "resourceId",
            "targetRole",
            "responseLanguage",
            "createdAt",
        )
        if hasattr(context, key) and getattr(context, key) is not None
    }


def _context_field(context: Any, field_name: str) -> Any:
    if isinstance(context, dict):
        return context.get(field_name)
    return getattr(context, field_name)


def _generation_id(context: Any) -> str:
    if isinstance(context, dict):
        return str(context.get("generationId") or context.get("taskId") or "report-generation")
    return str(
        getattr(context, "generationId", None)
        or getattr(context, "taskId", None)
        or "report-generation"
    )


def _build_mock_report_generation_output(task: Any) -> ReportGenerationOutput:
    if _context_field(task, "responseLanguage") == "zh":
        markdown = "\n".join(
            [
                "## 模拟面试报告",
                "",
                "### 总体评价",
                "候选人回答与岗位目标基本相关，后续报告会结合完整评分结果细化。",
                "",
                "### 逐题点评",
                "- 得分：7/10",
                "- 点评：回答具备基本方向，建议补充机制、边界和项目证据。",
            ]
        )
        comment = "回答具备基本方向，建议补充机制、边界和项目证据。"
    else:
        markdown = "\n".join(
            [
                "## Mock Interview Report",
                "",
                "### Overall Assessment",
                "The candidate's answers are generally relevant. The final report should "
                "be refined with complete evaluation results.",
            ]
        )
        comment = "The answer is relevant and should be strengthened with evidence."

    return ReportGenerationOutput.model_validate(
        {
            "summary": {
                "overallScore": 7,
                "overallComment": comment,
                "strengths": ["回答与岗位目标相关。"],
                "improvementPriorities": ["补充机制细节、边界条件和项目证据。"],
            },
            "questionReviews": [
                {
                    "questionId": "mock-question",
                    "attemptId": "mock-attempt",
                    "targetType": "main-question",
                    "question": "mock question",
                    "score": 7,
                    "comment": comment,
                    "missingPoints": [],
                    "improvementAdvice": ["补充更具体的项目证据。"],
                }
            ],
            "markdown": markdown,
        }
    )
