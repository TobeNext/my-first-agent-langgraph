from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.domain.question_metadata import format_skill_area, normalize_skill_area_from_text
from app.domain.question_planner import ProfessionalQuestionPlan
from app.domain.question_query import (
    RetrievalQueryIntent,
    build_jd_only_requeries,
    build_professional_requeries,
    build_professional_skill_query,
    build_project_experience_query,
    build_resume_jd_match_requeries,
    build_resume_only_requeries,
    describe_professional_plan_skill,
)
from app.domain.resume_jd_match import ResumeJdMatchAnalysis
from app.integrations.embeddings import embed_query_text
from app.integrations.keyword_question_store import KeywordQuestionStore
from app.integrations.milvus_store import MilvusQuestionStore
from app.schemas.interview_state import InterviewQuestionCandidate, RoundType

VECTOR_RECALL_TOP_K = 25
BM25_RECALL_TOP_K = 25
RRF_MERGE_TOP_K = 30
RERANK_TOP_K = 5
RRF_K = 60


@dataclass(frozen=True)
class RagRecallTrace:
    timestamp: str
    roundType: RoundType
    skill: str
    queryText: str
    logContext: str
    candidateQuestionIds: list[str]
    selectedQuestionIds: list[str]
    candidates: list[dict]
    finalSelectedQuestions: list[dict]


@dataclass(frozen=True)
class QueryInterviewQuestionsResult:
    count: int
    questions: list[InterviewQuestionCandidate]
    bm25Candidates: list[InterviewQuestionCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class RetrieveInitializationQuestionsResult:
    professionalQuestions: list[InterviewQuestionCandidate]
    projectQuestions: list[InterviewQuestionCandidate]
    recallTraces: list[RagRecallTrace] = field(default_factory=list)


@dataclass
class QuestionSelectionContext:
    selectedQuestionIds: set[str] = field(default_factory=set)
    selectedQuestionTexts: list[str] = field(default_factory=list)


def retrieve_initialization_questions(
    *,
    selected_direction: str,
    raw_kickoff_message: str,
    professional_skills: str,
    normalized_professional_skills: list[str],
    project_experience: str,
    normalized_project_topics: list[str],
    job_description: str,
    professional_question_plan: list[ProfessionalQuestionPlan],
    match_analysis: ResumeJdMatchAnalysis | None = None,
    store: MilvusQuestionStore | None = None,
    keyword_store: KeywordQuestionStore | None = None,
) -> RetrieveInitializationQuestionsResult:
    traces: list[RagRecallTrace] = []
    runtime_store = store or MilvusQuestionStore()
    professional: list[InterviewQuestionCandidate] = []
    selection_context = QuestionSelectionContext()

    if match_analysis is not None and job_description.strip():
        professional.extend(
            _retrieve_professional_questions_from_match_analysis(
                selected_direction=selected_direction,
                match_analysis=match_analysis,
                desired_count=len(professional_question_plan),
                store=runtime_store,
                keyword_store=keyword_store,
                selection_context=selection_context,
                traces=traces,
            )
        )
    elif professional_question_plan:
        for plan in professional_question_plan:
            skill = describe_professional_plan_skill(plan)
            query = build_professional_skill_query(
                selected_direction=selected_direction,
                plan=plan,
                professional_skills=professional_skills,
                project_experience=project_experience,
                normalized_skills=normalized_professional_skills,
            )
            requery_intents = build_professional_requeries(
                selected_direction=selected_direction,
                plan=plan,
                professional_skills=professional_skills,
                project_experience=project_experience,
                normalized_skills=normalized_professional_skills,
            )
            result = query_questions_multi(
                query_intents=requery_intents,
                top_k=1,
                round_type="professional-skills",
                skill=skill,
                store=runtime_store,
                keyword_store=keyword_store,
            )
            duplicate_ids = set(selection_context.selectedQuestionIds)
            duplicate_texts = list(selection_context.selectedQuestionTexts)
            selected = _select_novel_questions(result.questions, selection_context, top_k=1)
            professional.extend(selected)
            traces.append(
                _trace(
                    "professional-skills",
                    skill,
                    query,
                    f"initialization:professional-skills:{skill}",
                    result.bm25Candidates,
                    selected,
                    duplicate_question_ids=duplicate_ids,
                    duplicate_question_texts=duplicate_texts,
                )
            )
    else:
        query = _round_context_query(
            selected_direction=selected_direction,
            round_type="professional-skills",
            section_content=professional_skills,
            raw_kickoff_message=raw_kickoff_message,
        )
        result = query_questions(
            query_text=query,
            top_k=1,
            round_type="professional-skills",
            skill="professional-skills-context",
            store=runtime_store,
        )
        duplicate_ids = set(selection_context.selectedQuestionIds)
        duplicate_texts = list(selection_context.selectedQuestionTexts)
        selected = _select_novel_questions(result.questions, selection_context, top_k=1)
        professional.extend(selected)
        traces.append(
            _trace(
                "professional-skills",
                "professional-skills-context",
                query,
                "initialization:professional-skills:context",
                result.bm25Candidates,
                selected,
                duplicate_question_ids=duplicate_ids,
                duplicate_question_texts=duplicate_texts,
            )
        )

    project_query = build_project_experience_query(
        selected_direction=selected_direction,
        project_experience=project_experience,
        raw_kickoff_message=raw_kickoff_message,
        normalized_project_topics=normalized_project_topics,
        job_description=job_description,
    )
    project_result = query_questions(
        query_text=project_query,
        top_k=1,
        round_type="project-experience",
        skill="project-experience-context",
        store=runtime_store,
    )
    traces.append(
        _trace(
            "project-experience",
            "project-experience-context",
            project_query,
            "initialization:project-experience:context",
            project_result.bm25Candidates,
            project_result.questions,
        )
    )

    return RetrieveInitializationQuestionsResult(
        professionalQuestions=professional,
        projectQuestions=project_result.questions,
        recallTraces=traces,
    )


def _retrieve_professional_questions_from_match_analysis(
    *,
    selected_direction: str,
    match_analysis: ResumeJdMatchAnalysis,
    desired_count: int,
    store: MilvusQuestionStore,
    keyword_store: KeywordQuestionStore | None,
    selection_context: QuestionSelectionContext,
    traces: list[RagRecallTrace],
) -> list[InterviewQuestionCandidate]:
    if desired_count <= 0:
        return []
    professional: list[InterviewQuestionCandidate] = []
    for index, item in enumerate(match_analysis.resumeJdMatch, start=1):
        remaining = desired_count - len(professional)
        if remaining <= 0:
            break
        skill = f"resume-jd-match:{item.resumeSignal}"
        query_intents = build_resume_jd_match_requeries(
            selected_direction=selected_direction,
            item=item,
        )
        result = query_questions_multi(
            query_intents=query_intents,
            top_k=remaining,
            round_type="professional-skills",
            skill=skill,
            store=store,
            keyword_store=keyword_store,
        )
        duplicate_ids = set(selection_context.selectedQuestionIds)
        duplicate_texts = list(selection_context.selectedQuestionTexts)
        selected = _select_novel_questions(
            result.questions,
            selection_context,
            top_k=remaining,
        )
        professional.extend(selected)
        traces.append(
            _trace(
                "professional-skills",
                skill,
                "\n\n".join(intent.query for intent in query_intents),
                f"initialization:professional-skills:resume-jd-match:{index}",
                result.bm25Candidates,
                selected,
                duplicate_question_ids=duplicate_ids,
                duplicate_question_texts=duplicate_texts,
            )
        )

    if match_analysis.resumeOnly and len(professional) < desired_count:
        item = match_analysis.resumeOnly[0]
        skill = f"resume-only:{item.resumeSignal}"
        query_intents = build_resume_only_requeries(
            selected_direction=selected_direction,
            item=item,
        )
        result = query_questions_multi(
            query_intents=query_intents,
            top_k=1,
            round_type="professional-skills",
            skill=skill,
            store=store,
            keyword_store=keyword_store,
        )
        duplicate_ids = set(selection_context.selectedQuestionIds)
        duplicate_texts = list(selection_context.selectedQuestionTexts)
        selected = _select_novel_questions(result.questions, selection_context, top_k=1)
        professional.extend(selected)
        traces.append(
            _trace(
                "professional-skills",
                skill,
                "\n\n".join(intent.query for intent in query_intents),
                "initialization:professional-skills:resume-only",
                result.bm25Candidates,
                selected,
                duplicate_question_ids=duplicate_ids,
                duplicate_question_texts=duplicate_texts,
            )
        )

    if match_analysis.jdOnly and len(professional) < desired_count:
        item = match_analysis.jdOnly[0]
        skill = f"jd-only:{item.jobSignal}"
        query_intents = build_jd_only_requeries(
            selected_direction=selected_direction,
            item=item,
        )
        result = query_questions_multi(
            query_intents=query_intents,
            top_k=1,
            round_type="professional-skills",
            skill=skill,
            store=store,
            keyword_store=keyword_store,
        )
        duplicate_ids = set(selection_context.selectedQuestionIds)
        duplicate_texts = list(selection_context.selectedQuestionTexts)
        selected = _select_novel_questions(result.questions, selection_context, top_k=1)
        professional.extend(selected)
        traces.append(
            _trace(
                "professional-skills",
                skill,
                "\n\n".join(intent.query for intent in query_intents),
                "initialization:professional-skills:jd-only",
                result.bm25Candidates,
                selected,
                duplicate_question_ids=duplicate_ids,
                duplicate_question_texts=duplicate_texts,
            )
        )
    return professional


def query_questions(
    *,
    query_text: str,
    top_k: int,
    round_type: RoundType,
    skill: str,
    store: MilvusQuestionStore,
    keyword_store: KeywordQuestionStore | None = None,
) -> QueryInterviewQuestionsResult:
    with _get_tracer().start_as_current_span(
        "rag.question_retrieval.query",
        attributes=_retrieval_span_attributes(
            round_type=round_type,
            skill=skill,
            top_k=top_k,
            query_count=1,
        ),
    ) as span:
        try:
            vector = embed_query_text(query_text)
            vector_hits = store.search(
                vector=vector,
                top_k=max(VECTOR_RECALL_TOP_K, top_k),
                round_type=round_type,
            ).questions
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            vector_hits = []

        # Independent BM25 keyword retrieval from the full collection
        bm25_hits_raw: list[InterviewQuestionCandidate] = []
        if keyword_store is not None:
            try:
                keyword_result = keyword_store.search(
                    query_text=query_text,
                    top_k=BM25_RECALL_TOP_K,
                    round_type=round_type,
                )
                bm25_hits_raw = keyword_result.questions[:BM25_RECALL_TOP_K]
            except Exception as exc:
                span.record_exception(exc)
                bm25_hits_raw = []
        else:
            # Fallback: local BM25 rescore of vector hits (legacy behavior)
            bm25_hits_raw = bm25_rerank_questions(vector_hits, query_text=query_text)

        bm25_hits = bm25_hits_raw[:BM25_RECALL_TOP_K]
        ranked_lists = [vector_hits[:VECTOR_RECALL_TOP_K], bm25_hits]
        rrf_scores = _rrf_rank_scores(ranked_lists)
        fused = _rrf_merge_ranked_candidates(ranked_lists, rrf_scores=rrf_scores)[
            :RRF_MERGE_TOP_K
        ]
        duplicate_veto_ids = _duplicate_veto_question_ids(fused)
        deduped = [
            question for question in fused if question.id not in duplicate_veto_ids
        ]
        reranked = metadata_rerank_questions(
            deduped,
            query_text=query_text,
            rrf_scores=rrf_scores,
        )[:RERANK_TOP_K]
        bm25_candidates = metadata_rerank_questions(
            fused,
            query_text=query_text,
            rrf_scores=rrf_scores,
            duplicate_veto_ids=duplicate_veto_ids,
        )[:RERANK_TOP_K]
        questions = _sample_questions(reranked, top_k)
        span.set_attribute("rag.candidate_count", sum(len(items) for items in ranked_lists))
        span.set_attribute("rag.fused_count", len(fused))
        span.set_attribute("rag.duplicate_veto_count", len(duplicate_veto_ids))
        span.set_attribute("rag.reranked_count", len(reranked))
        span.set_attribute("rag.result_count", len(questions))
        return QueryInterviewQuestionsResult(
            count=len(questions),
            questions=questions,
            bm25Candidates=bm25_candidates,
        )


def query_questions_multi(
    *,
    query_intents: list[RetrievalQueryIntent],
    top_k: int,
    round_type: RoundType,
    skill: str,
    store: MilvusQuestionStore,
    keyword_store: KeywordQuestionStore | None = None,
) -> QueryInterviewQuestionsResult:
    if not query_intents:
        return QueryInterviewQuestionsResult(count=0, questions=[], bm25Candidates=[])

    with _get_tracer().start_as_current_span(
        "rag.question_retrieval.multi_query",
        attributes=_retrieval_span_attributes(
            round_type=round_type,
            skill=skill,
            top_k=top_k,
            query_count=len(query_intents),
        ),
    ) as span:
        ranked_lists: list[list[InterviewQuestionCandidate]] = []
        vector_count = 0
        keyword_count = 0
        for intent in query_intents:
            try:
                vector = embed_query_text(intent.query)
                vector_hits = store.search(
                    vector=vector,
                    top_k=max(VECTOR_RECALL_TOP_K, top_k),
                    round_type=round_type,
                ).questions[:VECTOR_RECALL_TOP_K]
                ranked_lists.append(vector_hits)
                vector_count += len(vector_hits)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                ranked_lists.append([])
            if keyword_store is None:
                continue
            try:
                keyword_hits = keyword_store.search(
                    query_text=intent.query,
                    top_k=max(BM25_RECALL_TOP_K, top_k),
                    round_type=round_type,
                ).questions[:BM25_RECALL_TOP_K]
                ranked_lists.append(keyword_hits)
                keyword_count += len(keyword_hits)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                ranked_lists.append([])

        query_text = "\n\n".join(f"[{intent.type}]\n{intent.query}" for intent in query_intents)
        rrf_scores = _rrf_rank_scores(ranked_lists)
        fused = _rrf_merge_ranked_candidates(ranked_lists, rrf_scores=rrf_scores)[
            :RRF_MERGE_TOP_K
        ]
        duplicate_veto_ids = _duplicate_veto_question_ids(fused)
        deduped = [
            question for question in fused if question.id not in duplicate_veto_ids
        ]
        reranked = metadata_rerank_questions(
            deduped,
            query_text=query_text,
            rrf_scores=rrf_scores,
        )[:RERANK_TOP_K]
        bm25_candidates = metadata_rerank_questions(
            fused,
            query_text=query_text,
            rrf_scores=rrf_scores,
            duplicate_veto_ids=duplicate_veto_ids,
        )[:RERANK_TOP_K]
        questions = _sample_questions(reranked, top_k)
        span.set_attribute("rag.candidate_count", sum(len(items) for items in ranked_lists))
        span.set_attribute("rag.vector_candidate_count", vector_count)
        span.set_attribute("rag.keyword_candidate_count", keyword_count)
        span.set_attribute("rag.fused_count", len(fused))
        span.set_attribute("rag.duplicate_veto_count", len(duplicate_veto_ids))
        span.set_attribute("rag.reranked_count", len(reranked))
        span.set_attribute("rag.result_count", len(questions))
        return QueryInterviewQuestionsResult(
            count=len(questions),
            questions=questions,
            bm25Candidates=bm25_candidates,
        )


def hybrid_rerank_questions(
    candidates: list[InterviewQuestionCandidate],
    *,
    query_text: str,
) -> list[InterviewQuestionCandidate]:
    scored = _build_bm25_scored_candidates(candidates, query_text)
    return [entry["question"] for entry in scored]


def metadata_rerank_questions(
    candidates: list[InterviewQuestionCandidate],
    *,
    query_text: str,
    rrf_scores: dict[str, float] | None = None,
    duplicate_veto_ids: set[str] | None = None,
) -> list[InterviewQuestionCandidate]:
    scored = _build_metadata_scored_candidates(
        candidates,
        query_text,
        rrf_scores=rrf_scores,
        duplicate_veto_ids=duplicate_veto_ids,
    )
    return [
        entry["question"].model_copy(update={"selectionScore": entry["hybridScore"]})
        for entry in scored
    ]


def bm25_rerank_questions(
    candidates: list[InterviewQuestionCandidate],
    *,
    query_text: str,
) -> list[InterviewQuestionCandidate]:
    scored = _build_bm25_scored_candidates(candidates, query_text)
    return [entry["question"] for entry in scored]


def extract_jd_skill_area(query_text: str) -> list[str]:
    skills = normalize_skill_area_from_text(query_text)
    return [] if skills == ["agent"] else skills


def _rrf_merge_ranked_candidates(
    ranked_lists: list[list[InterviewQuestionCandidate]],
    *,
    rrf_scores: dict[str, float] | None = None,
) -> list[InterviewQuestionCandidate]:
    by_id: dict[str, InterviewQuestionCandidate] = {}
    scores = rrf_scores or _rrf_rank_scores(ranked_lists)

    for ranked in ranked_lists:
        for question in ranked:
            by_id.setdefault(question.id, question)

    return sorted(
        by_id.values(),
        key=lambda question: (scores.get(question.id, 0), question.score, question.id),
        reverse=True,
    )


def _rrf_rank_scores(
    ranked_lists: list[list[InterviewQuestionCandidate]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, question in enumerate(ranked, start=1):
            scores[question.id] = scores.get(question.id, 0.0) + 1 / (RRF_K + rank)
    return scores


def _trace(
    round_type: RoundType,
    skill: str,
    query: str,
    log_context: str,
    candidates: list[InterviewQuestionCandidate],
    selected_questions: list[InterviewQuestionCandidate],
    duplicate_question_ids: set[str] | None = None,
    duplicate_question_texts: list[str] | None = None,
) -> RagRecallTrace:
    duplicate_veto_ids = _duplicate_veto_question_ids(candidates)
    scored_candidates = _build_metadata_scored_candidates(
        candidates,
        query,
        duplicate_question_ids=duplicate_question_ids,
        duplicate_question_texts=duplicate_question_texts,
        duplicate_veto_ids=duplicate_veto_ids,
    )
    selected_ids = {question.id for question in selected_questions}
    selected_entries = [
        entry for entry in scored_candidates if entry["question"].id in selected_ids
    ]
    return RagRecallTrace(
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        roundType=round_type,
        skill=skill,
        queryText=query,
        logContext=log_context,
        candidateQuestionIds=[question.id for question in candidates],
        selectedQuestionIds=[question.id for question in selected_questions],
        candidates=[
            _trace_candidate(entry, index + 1)
            for index, entry in enumerate(scored_candidates)
        ],
        finalSelectedQuestions=[
            _trace_selection(entry, index + 1)
            for index, entry in enumerate(selected_entries)
        ],
    )


def _build_bm25_scored_candidates(
    candidates: list[InterviewQuestionCandidate],
    query_text: str,
) -> list[dict]:
    query_terms = _tokenize_for_bm25(query_text)
    document_terms = [_tokenize_for_bm25(_candidate_bm25_text(question)) for question in candidates]
    average_length = (
        sum(len(terms) for terms in document_terms) / len(document_terms)
        if document_terms
        else 0
    )
    document_frequency = Counter(
        term for terms in document_terms for term in set(terms)
    )
    entries: list[dict] = []
    for question, terms in zip(candidates, document_terms, strict=True):
        vector_score = float(question.score)
        bm25_score = _bm25_score(
            query_terms=query_terms,
            document_terms=terms,
            document_frequency=document_frequency,
            document_count=len(document_terms),
            average_document_length=average_length,
        )
        entries.append(
            {
                "question": question,
                "vectorScore": vector_score,
                "bm25Score": bm25_score,
                "hybridScore": bm25_score,
                "matchedSkillArea": _matched_query_skills(question, query_text),
            }
        )
    entries.sort(
        key=lambda entry: (entry["bm25Score"], entry["vectorScore"], entry["question"].id),
        reverse=True,
    )
    return entries


def _build_metadata_scored_candidates(
    candidates: list[InterviewQuestionCandidate],
    query_text: str,
    duplicate_question_ids: set[str] | None = None,
    duplicate_question_texts: list[str] | None = None,
    rrf_scores: dict[str, float] | None = None,
    duplicate_veto_ids: set[str] | None = None,
) -> list[dict]:
    bm25_entries = _build_bm25_scored_candidates(candidates, query_text)
    if not bm25_entries:
        return []

    raw_rrf_scores = rrf_scores or {
        entry["question"].id: float(len(bm25_entries) - index)
        for index, entry in enumerate(bm25_entries)
    }
    rrf_values = [raw_rrf_scores.get(entry["question"].id, 0.0) for entry in bm25_entries]
    min_rrf = min(rrf_values)
    max_rrf = max(rrf_values)

    scored: list[dict] = []
    for entry in bm25_entries:
        question: InterviewQuestionCandidate = entry["question"]
        rrf_score = raw_rrf_scores.get(question.id, 0.0)
        rrf_score_norm = _normalize_score(rrf_score, min_rrf, max_rrf)
        question_type_score = _question_type_score(question.questionType)
        is_duplicate = _is_duplicate_question(
            question,
            duplicate_question_ids or set(),
            duplicate_question_texts or [],
        ) or question.id in (duplicate_veto_ids or set())
        score_breakdown = {
            "rrf": round(rrf_score_norm, 4),
            "questionType": round(question_type_score, 4),
        }
        hybrid_score = 0.9 * rrf_score_norm + 0.1 * question_type_score
        scored.append(
            {
                **entry,
                "rrfScore": rrf_score,
                "questionTypeScore": question_type_score,
                "hybridScore": hybrid_score,
                "scoreBreakdown": score_breakdown,
                "matchedMetadata": _matched_metadata(question, query_text),
                "isDuplicate": is_duplicate,
                "filterReason": (
                    "duplicate-veto"
                    if question.id in (duplicate_veto_ids or set())
                    else None
                ),
            }
        )

    scored.sort(
        key=lambda entry: (
            entry["hybridScore"],
            entry["rrfScore"],
            entry["questionTypeScore"],
            entry["bm25Score"],
            entry["vectorScore"],
            entry["question"].id,
        ),
        reverse=True,
    )
    return scored


def _normalize_score(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 1.0
    return (value - min_value) / (max_value - min_value)


def _metadata_skill_match_score(question: InterviewQuestionCandidate, query_text: str) -> float:
    query_skills = extract_jd_skill_area(query_text)
    if not query_skills:
        return 0.0
    return len(_matched_skill_area(question, query_skills)) / len(query_skills)


def _metadata_job_match_score(question: InterviewQuestionCandidate, query_text: str) -> float:
    candidate_tokens = set(_tokenize_for_bm25(_candidate_bm25_text(question)))
    job_lines = [
        line
        for line in query_text.splitlines()
        if any(
            label in line.lower()
            for label in ["target role", "job ", "responsibility", "driver"]
        )
    ]
    job_tokens = set(_tokenize_for_bm25("\n".join(job_lines)))
    if not candidate_tokens or not job_tokens:
        return 0.0
    return len(candidate_tokens & job_tokens) / len(job_tokens)


def _question_type_score(question_type: str | None) -> float:
    normalized = (question_type or "").strip().lower().replace("-", "_")
    if normalized == "system_design":
        return 1.0
    if normalized == "technical":
        return 0.9
    if normalized in {"experience_probe", "case_analysis", "scenario"}:
        return 0.85
    if normalized == "knowledge_check":
        return 0.7
    if normalized == "culture_fit":
        return 0.3
    return 0.5


def _difficulty_match_score(difficulty: str | None, query_text: str) -> float:
    normalized = (difficulty or "").strip().lower()
    expects_hard = any(value in query_text.lower() for value in ["hard", "senior", "scenario"])
    if expects_hard:
        return 1.0 if normalized in {"hard", "senior"} else 0.65
    return 1.0 if normalized in {"medium", "middle", "mid"} else 0.75


def _sample_questions(
    candidates: list[InterviewQuestionCandidate],
    top_k: int,
) -> list[InterviewQuestionCandidate]:
    return _weighted_sample_questions(candidates, top_k)


def _weighted_sample_questions(
    candidates: list[InterviewQuestionCandidate],
    top_k: int,
    *,
    random_source: random.Random | None = None,
) -> list[InterviewQuestionCandidate]:
    if not candidates or top_k <= 0:
        return []
    rng = random_source or random
    remaining = list(candidates[:RERANK_TOP_K])
    selected: list[InterviewQuestionCandidate] = []
    while remaining and len(selected) < top_k:
        weights = [_candidate_selection_weight(question) for question in remaining]
        total_weight = sum(weights)
        if total_weight <= 0:
            selected.append(remaining.pop(0))
            continue
        marker = rng.random() * total_weight
        cumulative = 0.0
        selected_index = 0
        for index, weight in enumerate(weights):
            cumulative += weight
            if marker <= cumulative:
                selected_index = index
                break
        selected.append(remaining.pop(selected_index))
    return selected


def _candidate_selection_weight(question: InterviewQuestionCandidate) -> float:
    score = getattr(question, "selectionScore", None)
    if isinstance(score, int | float):
        return max(float(score), 1e-6)
    return max(float(question.score), 1e-6)


def _duplicate_veto_question_ids(
    candidates: list[InterviewQuestionCandidate],
) -> set[str]:
    vetoed: set[str] = set()
    kept: list[InterviewQuestionCandidate] = []
    normalized_questions: set[str] = set()
    for question in candidates:
        normalized = _normalize_question_text(question.text)
        if normalized and normalized in normalized_questions:
            vetoed.add(question.id)
            continue
        if any(
            _text_overlap_score(question.text, kept_question.text) >= 0.82
            for kept_question in kept
        ):
            vetoed.add(question.id)
            continue
        kept.append(question)
        if normalized:
            normalized_questions.add(normalized)
    return vetoed


def _select_novel_questions(
    questions: list[InterviewQuestionCandidate],
    context: QuestionSelectionContext,
    *,
    top_k: int,
) -> list[InterviewQuestionCandidate]:
    selected: list[InterviewQuestionCandidate] = []
    for question in questions:
        if len(selected) >= top_k:
            break
        if _is_duplicate_question(
            question,
            context.selectedQuestionIds,
            context.selectedQuestionTexts,
        ):
            continue
        selected.append(question)
        context.selectedQuestionIds.add(question.id)
        context.selectedQuestionTexts.append(question.text)
    return selected


def _is_duplicate_question(
    question: InterviewQuestionCandidate,
    selected_ids: set[str],
    selected_texts: list[str],
) -> bool:
    if question.id in selected_ids:
        return True
    return any(
        _text_overlap_score(question.text, selected_text) >= 0.82
        for selected_text in selected_texts
    )


def _text_overlap_score(left: str, right: str) -> float:
    left_terms = set(_tokenize_for_bm25(left))
    right_terms = set(_tokenize_for_bm25(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def _normalize_question_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()


def _trace_candidate(entry: dict, final_selection_rank: int | None) -> dict:
    question: InterviewQuestionCandidate = entry["question"]
    return {
        "id": question.id,
        "questionText": question.text,
        "vectorScore": round(float(entry["vectorScore"]), 4),
        "bm25Score": round(float(entry["bm25Score"]), 4),
        "hybridScore": round(float(entry["hybridScore"]), 4),
        "matchedSkillArea": entry["matchedSkillArea"],
        "scoreBreakdown": entry.get("scoreBreakdown", {}),
        "matchedMetadata": entry.get("matchedMetadata", {}),
        "isDuplicate": bool(entry.get("isDuplicate", False)),
        "rerankRank": final_selection_rank,
        "finalSelectionRank": final_selection_rank,
        "filterReason": entry.get("filterReason")
        or ("selected" if final_selection_rank is not None else "not-selected"),
    }


def _trace_selection(entry: dict, final_selection_rank: int) -> dict:
    question: InterviewQuestionCandidate = entry["question"]
    return {
        "id": question.id,
        "questionText": question.text,
        "vectorScore": round(float(entry["vectorScore"]), 4),
        "bm25Score": round(float(entry["bm25Score"]), 4),
        "hybridScore": round(float(entry["hybridScore"]), 4),
        "matchedSkillArea": entry["matchedSkillArea"],
        "scoreBreakdown": entry.get("scoreBreakdown", {}),
        "matchedMetadata": entry.get("matchedMetadata", {}),
        "isDuplicate": bool(entry.get("isDuplicate", False)),
        "rerankRank": final_selection_rank,
        "finalSelectionRank": final_selection_rank,
    }


def _skill_match_score(question: InterviewQuestionCandidate, jd_skill_area: list[str]) -> float:
    if not jd_skill_area:
        return 0
    return len(_matched_skill_area(question, jd_skill_area)) / len(jd_skill_area)


def _matched_skill_area(
    question: InterviewQuestionCandidate,
    jd_skill_area: list[str],
) -> list[str]:
    question_skills = [skill.lower() for skill in format_skill_area(question.skillArea)]
    return [skill for skill in jd_skill_area if skill.lower() in question_skills]


def _matched_query_skills(
    question: InterviewQuestionCandidate,
    query_text: str,
) -> list[str]:
    query_skills = extract_jd_skill_area(query_text)
    return _matched_skill_area(question, query_skills)


def _matched_metadata(question: InterviewQuestionCandidate, query_text: str) -> dict:
    return {
        "skills": _matched_query_skills(question, query_text),
        "jobDuties": _matched_job_duties(question, query_text),
        "questionType": question.questionType,
        "level": question.level or question.difficulty,
    }


def _matched_job_duties(question: InterviewQuestionCandidate, query_text: str) -> list[str]:
    duties = question.jobDuties or []
    if not duties:
        return []
    query_tokens = set(_tokenize_for_bm25(query_text))
    return [
        duty
        for duty in duties
        if query_tokens and query_tokens & set(_tokenize_for_bm25(duty))
    ]


def _candidate_bm25_text(question: InterviewQuestionCandidate) -> str:
    return "\n".join(
        [
            question.text,
            question.answer or "",
            question.tags or "",
            " ".join(format_skill_area(question.skillArea)),
        ]
    )


def _tokenize_for_bm25(value: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"[a-zA-Z0-9_.+#-]+|[\u3400-\u9fff]+", value.lower()):
        text = match.group(0).strip()
        if not text:
            continue
        if re.fullmatch(r"[\u3400-\u9fff]+", text):
            tokens.extend(_chinese_bigrams(text))
        else:
            tokens.append(text)
    return tokens


def _chinese_bigrams(value: str) -> list[str]:
    if len(value) <= 2:
        return [value]
    return [value[index : index + 2] for index in range(len(value) - 1)]


def _bm25_score(
    *,
    query_terms: list[str],
    document_terms: list[str],
    document_frequency: Counter[str],
    document_count: int,
    average_document_length: float,
) -> float:
    if not query_terms or not document_terms or document_count <= 0:
        return 0

    k1 = 1.5
    b = 0.75
    term_frequency = Counter(document_terms)
    document_length = len(document_terms)
    score = 0.0
    for term in set(query_terms):
        frequency = term_frequency.get(term, 0)
        if frequency <= 0:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        denominator = frequency + k1 * (
            1 - b + b * document_length / (average_document_length or 1)
        )
        score += idf * (frequency * (k1 + 1)) / denominator
    return score


def _retrieval_span_attributes(
    *,
    round_type: RoundType,
    skill: str,
    top_k: int,
    query_count: int,
) -> dict[str, str | int]:
    return {
        "interview.round_type": round_type,
        "interview.skill": skill,
        "rag.top_k": top_k,
        "rag.query_count": query_count,
    }


def _get_tracer() -> trace.Tracer:
    return trace.get_tracer("interview-python-agent")


def _round_context_query(
    *,
    selected_direction: str,
    round_type: RoundType,
    section_content: str,
    raw_kickoff_message: str,
) -> str:
    heading = "Professional skills" if round_type == "professional-skills" else "Project experience"
    return "\n".join(
        [
            f"Target role: {selected_direction}",
            f"Round type: {round_type}",
            f"{heading} context:",
            section_content.strip() or raw_kickoff_message,
        ]
    )
