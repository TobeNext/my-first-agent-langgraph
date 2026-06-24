from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from tests.evals.evaluators.deepseek_judge import create_deepeval_judge, has_eval_judge_key
from tests.evals.run_interview_eval_target import load_interview_cases, run_cases

DeepEvalScope = Literal["follow_up_generation", "report_generation"]
MetricTier = Literal["required", "optional", "custom_geval"]


@dataclass(frozen=True)
class DeepEvalMetricSpec:
    scope: DeepEvalScope
    tier: MetricTier
    name: str
    metric_class: str
    reason: str


@dataclass(frozen=True)
class InterviewQualityCase:
    case_id: str
    input_text: str
    actual_output: str
    expected_output: str
    forbidden_fragments: tuple[str, ...]
    required_fragments: tuple[str, ...]
    threshold: float = 0.75


DEEPEVAL_METRIC_SPECS = (
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="required",
        name="faithfulness",
        metric_class="FaithfulnessMetric",
        reason="Ground follow-up questions in the resume, current question, and previous answer.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="required",
        name="hallucination",
        metric_class="HallucinationMetric",
        reason="Detect invented facts or assumptions in generated follow-up questions.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="required",
        name="bias",
        metric_class="BiasMetric",
        reason="Prevent discriminatory interview wording or unfair demographic assumptions.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="required",
        name="toxicity",
        metric_class="ToxicityMetric",
        reason="Keep interviewer tone professional and non-abusive.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="optional",
        name="answer_relevancy",
        metric_class="AnswerRelevancyMetric",
        reason="Check whether the follow-up stays on the current interview thread.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="optional",
        name="contextual_relevancy",
        metric_class="ContextualRelevancyMetric",
        reason="Check whether the follow-up uses the provided context effectively.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="custom_geval",
        name="followup_specificity",
        metric_class="GEval",
        reason="Prefer specific, guided follow-ups over broad prompts like 'please explain more'.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="custom_geval",
        name="followup_non_repetition",
        metric_class="GEval",
        reason="Avoid repeating information the candidate already provided.",
    ),
    DeepEvalMetricSpec(
        scope="follow_up_generation",
        tier="custom_geval",
        name="followup_logical_coherence",
        metric_class="GEval",
        reason="Require a natural progression from the main question and previous answer.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="required",
        name="faithfulness",
        metric_class="FaithfulnessMetric",
        reason="Ensure report statements are supported by interview evidence.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="required",
        name="hallucination",
        metric_class="HallucinationMetric",
        reason="Prevent fabricated skills, projects, scores, or candidate claims.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="required",
        name="summarization",
        metric_class="SummarizationMetric",
        reason="Check report alignment with source evidence and coverage of key interview content.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="required",
        name="bias",
        metric_class="BiasMetric",
        reason="Prevent biased or unfair report comments.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="required",
        name="prompt_alignment",
        metric_class="PromptAlignmentMetric",
        reason="Ensure the report follows the required report structure and instructions.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="optional",
        name="task_completion",
        metric_class="TaskCompletionMetric",
        reason="Check that the report completes required conclusions and recommendations.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="optional",
        name="answer_relevancy",
        metric_class="AnswerRelevancyMetric",
        reason="Check that the report remains relevant to the report-generation request.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="custom_geval",
        name="report_dimension_coverage",
        metric_class="GEval",
        reason=(
            "Require coverage of communication, technical depth, problem solving, "
            "and learning ability."
        ),
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="custom_geval",
        name="report_score_consistency",
        metric_class="GEval",
        reason="Ensure scores are consistent with observed answer quality and missing points.",
    ),
    DeepEvalMetricSpec(
        scope="report_generation",
        tier="custom_geval",
        name="report_actionable_advice",
        metric_class="GEval",
        reason="Prefer specific and actionable improvement advice over generic suggestions.",
    ),
)


def test_deepeval_metric_pool_matches_followup_and_report_requirements() -> None:
    followup_required = _metric_names("follow_up_generation", "required")
    followup_optional = _metric_names("follow_up_generation", "optional")
    followup_custom = _metric_names("follow_up_generation", "custom_geval")
    report_required = _metric_names("report_generation", "required")
    report_optional = _metric_names("report_generation", "optional")
    report_custom = _metric_names("report_generation", "custom_geval")

    assert followup_required == {"faithfulness", "hallucination", "bias", "toxicity"}
    assert followup_optional == {"answer_relevancy", "contextual_relevancy"}
    assert followup_custom == {
        "followup_specificity",
        "followup_non_repetition",
        "followup_logical_coherence",
    }
    assert report_required == {
        "faithfulness",
        "hallucination",
        "summarization",
        "bias",
        "prompt_alignment",
    }
    assert report_optional == {"task_completion", "answer_relevancy"}
    assert report_custom == {
        "report_dimension_coverage",
        "report_score_consistency",
        "report_actionable_advice",
    }


def test_deepeval_gate_uses_interview_eval_target_contract(tmp_path: Path) -> None:
    cases = load_interview_cases()[:1]
    quality_cases = _build_quality_cases(cases, work_dir=tmp_path)

    assert len(quality_cases) == 1
    quality_case = quality_cases[0]
    assert quality_case.case_id == cases[0]["case_id"]
    assert "phase=" in quality_case.actual_output
    assert "report_status=" in quality_case.actual_output
    assert "expected_stage_path=" in quality_case.expected_output


def test_deepeval_gate_rule_fallback_from_eval_target(tmp_path: Path) -> None:
    quality_cases = _build_quality_cases(load_interview_cases()[:1], work_dir=tmp_path)

    for quality_case in quality_cases:
        score = _rule_score(quality_case)
        assert score >= quality_case.threshold, {
            "metric": "rule_interview_quality",
            "score": score,
            "threshold": quality_case.threshold,
            "case": quality_case.case_id,
            "actual_output": quality_case.actual_output,
        }


def test_deepeval_llm_judge_scores_eval_target_output(tmp_path: Path) -> None:
    if importlib.util.find_spec("deepeval") is None:
        pytest.skip("deepeval is not installed; rule fallback gate already ran.")
    if not has_eval_judge_key():
        pytest.skip("No eval judge key is configured; rule fallback gate already ran.")

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    model = create_deepeval_judge()
    metrics = [
        *_build_followup_generation_metrics(model),
        *_build_report_generation_metrics(model),
    ]

    for quality_case in _build_quality_cases(load_interview_cases()[:1], work_dir=tmp_path):
        assert_test(
            LLMTestCase(
                input=quality_case.input_text,
                actual_output=quality_case.actual_output,
                expected_output=quality_case.expected_output,
                context=[quality_case.expected_output],
                retrieval_context=[quality_case.expected_output],
            ),
            metrics,
        )


def _build_quality_cases(
    cases: list[dict[str, Any]],
    *,
    work_dir: Path,
) -> list[InterviewQualityCase]:
    summary = run_cases(cases, include_trajectory=True, work_dir=work_dir)
    assert summary["failed"] == 0, summary

    quality_cases = []
    results_by_case_id = {result["case_id"]: result for result in summary["results"]}
    for case in cases:
        result = results_by_case_id[case["case_id"]]
        quality_cases.append(
            InterviewQualityCase(
                case_id=case["case_id"],
                input_text=_case_input_text(case),
                actual_output=_result_output_text(result),
                expected_output=_expected_output_text(case),
                forbidden_fragments=tuple(str(item) for item in case.get("must_not_claim", [])),
                required_fragments=_required_fragments(case),
            )
        )
    return quality_cases


def _build_followup_generation_metrics(model: Any) -> list[Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
        ToxicityMetric,
    )
    from deepeval.test_case import SingleTurnParams

    return [
        FaithfulnessMetric(threshold=0.7, model=model),
        HallucinationMetric(threshold=0.3, model=model),
        BiasMetric(threshold=0.3, model=model),
        ToxicityMetric(threshold=0.3, model=model),
        AnswerRelevancyMetric(threshold=0.7, model=model),
        ContextualRelevancyMetric(threshold=0.7, model=model),
        GEval(
            name="followup-specificity",
            criteria=(
                "Evaluate whether the generated follow-up is specific and guided, "
                "not a generic request to explain more."
            ),
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=model,
            threshold=0.7,
        ),
        GEval(
            name="followup-non-repetition",
            criteria=(
                "Evaluate whether the follow-up avoids repeating information already "
                "contained in the provided interview evidence."
            ),
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=model,
            threshold=0.7,
        ),
        GEval(
            name="followup-logical-coherence",
            criteria=(
                "Evaluate whether the follow-up naturally progresses from the main "
                "question and previous answer without an abrupt topic jump."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            model=model,
            threshold=0.7,
        ),
    ]


def _build_report_generation_metrics(model: Any) -> list[Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
        PromptAlignmentMetric,
        SummarizationMetric,
        TaskCompletionMetric,
    )
    from deepeval.test_case import SingleTurnParams

    return [
        FaithfulnessMetric(threshold=0.7, model=model),
        HallucinationMetric(threshold=0.3, model=model),
        SummarizationMetric(threshold=0.7, model=model),
        BiasMetric(threshold=0.3, model=model),
        PromptAlignmentMetric(
            prompt_instructions=[
                "Include structured technical assessment sections.",
                "Keep every claim grounded in interview evidence.",
                "Do not include biased or sensitive demographic assumptions.",
            ],
            threshold=0.7,
            model=model,
        ),
        TaskCompletionMetric(
            task="Generate a complete interview report with summary, scores, evidence, and advice.",
            threshold=0.7,
            model=model,
        ),
        AnswerRelevancyMetric(threshold=0.7, model=model),
        GEval(
            name="report-dimension-coverage",
            criteria=(
                "Evaluate whether the report covers communication, technical depth, "
                "problem solving, and learning ability with explicit comments."
            ),
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=model,
            threshold=0.7,
        ),
        GEval(
            name="report-score-consistency",
            criteria=(
                "Evaluate whether report scores are consistent with observed correct "
                "answers, missing points, and incorrect points."
            ),
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=model,
            threshold=0.7,
        ),
        GEval(
            name="report-actionable-advice",
            criteria=(
                "Evaluate whether improvement advice is specific and actionable, "
                "not generic advice such as 'practice more'."
            ),
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=model,
            threshold=0.7,
        ),
    ]


def _metric_names(scope: DeepEvalScope, tier: MetricTier) -> set[str]:
    return {
        spec.name
        for spec in DEEPEVAL_METRIC_SPECS
        if spec.scope == scope and spec.tier == tier
    }


def _case_input_text(case: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"case_id={case['case_id']}",
            f"settings={json.dumps(case['settings'], ensure_ascii=False, sort_keys=True)}",
            f"expected_required_skills={_json_list(case.get('expected_required_skills', []))}",
            f"rubric={json.dumps(case.get('rubric', {}), ensure_ascii=False, sort_keys=True)}",
        ]
    )


def _result_output_text(result: dict[str, Any]) -> str:
    snapshot = result.get("final_snapshot") or {}
    progress = snapshot.get("progress") or {}
    trajectory = result.get("trajectory") or []
    trajectory_events = [str(item.get("event")) for item in trajectory]
    trajectory_stages = [
        str(item.get("current_stage"))
        for item in trajectory
        if item.get("current_stage")
    ]
    return "\n".join(
        [
            f"case_id={result['case_id']}",
            f"phase={snapshot.get('phase')}",
            f"active_round_type={snapshot.get('activeRoundType')}",
            f"active_node_topic={snapshot.get('activeNodeTopic')}",
            f"final_report_ready={snapshot.get('finalReportReady')}",
            f"progress_current_stage={progress.get('currentStage')}",
            f"report_status={result.get('report_status')}",
            f"report_markdown_available={result.get('report_markdown_available')}",
            f"trajectory_events={_json_list(trajectory_events)}",
            f"trajectory_stages={_json_list(trajectory_stages)}",
            f"errors={_json_list(result.get('errors', []))}",
        ]
    )


def _expected_output_text(case: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"case_id={case['case_id']}",
            f"expected_stage_path={_json_list(case.get('expected_stage_path', []))}",
            f"expected_required_skills={_json_list(case.get('expected_required_skills', []))}",
            f"must_not_claim={_json_list(case.get('must_not_claim', []))}",
            "report_status_boundary=report database status API when the case reaches wrap-up",
            "sensitive_body_policy=do not expose resume, JD, answer, prompt, or report bodies",
        ]
    )


def _required_fragments(case: dict[str, Any]) -> tuple[str, ...]:
    fragments = ["errors=[]"]
    expected_stage_path = [str(item) for item in case.get("expected_stage_path", [])]
    if expected_stage_path:
        fragments.append(f"progress_current_stage={expected_stage_path[-1]}")
    if "wrap-up" in expected_stage_path:
        fragments.extend(
            [
                "phase=completed",
                "final_report_ready=True",
                "report_status=succeeded",
                "report_markdown_available=True",
            ]
        )
    return tuple(fragments)


def _rule_score(quality_case: InterviewQualityCase) -> float:
    output = quality_case.actual_output.lower()
    expected_hits = sum(
        1 for fragment in quality_case.required_fragments if fragment.lower() in output
    )
    forbidden_hits = sum(
        1 for fragment in quality_case.forbidden_fragments if fragment.lower() in output
    )
    expected_score = expected_hits / max(1, len(quality_case.required_fragments))
    penalty = forbidden_hits / max(1, len(quality_case.forbidden_fragments))
    return round(max(0.0, expected_score - penalty), 4)


def _json_list(values: Any) -> str:
    return json.dumps(list(values), ensure_ascii=False, sort_keys=True)
