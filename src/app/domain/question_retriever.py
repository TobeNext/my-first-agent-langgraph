from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.question_metadata import format_skill_area, normalize_skill_area_from_text
from app.domain.question_planner import ProfessionalQuestionPlan
from app.domain.question_query import (
    build_professional_skill_query,
    build_project_experience_query,
    describe_professional_plan_skill,
)
from app.integrations.embeddings import embed_query_text
from app.integrations.milvus_store import MilvusQuestionStore
from app.schemas.interview_state import InterviewQuestionCandidate, RoundType

VECTOR_RECALL_TOP_K = 20
BM25_RERANK_TOP_K = 5


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
    store: MilvusQuestionStore | None = None,
) -> RetrieveInitializationQuestionsResult:
    traces: list[RagRecallTrace] = []
    runtime_store = store or MilvusQuestionStore()
    professional: list[InterviewQuestionCandidate] = []

    if professional_question_plan:
        for plan in professional_question_plan:
            skill = describe_professional_plan_skill(plan)
            query = build_professional_skill_query(
                selected_direction=selected_direction,
                plan=plan,
                professional_skills=professional_skills,
                project_experience=project_experience,
                normalized_skills=normalized_professional_skills,
            )
            result = query_questions(
                query_text=query,
                top_k=1,
                round_type="professional-skills",
                skill=skill,
                store=runtime_store,
            )
            professional.extend(result.questions)
            traces.append(
                _trace(
                    "professional-skills",
                    skill,
                    query,
                    f"initialization:professional-skills:{skill}",
                    result.bm25Candidates,
                    result.questions,
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
        professional.extend(result.questions)
        traces.append(
            _trace(
                "professional-skills",
                "professional-skills-context",
                query,
                "initialization:professional-skills:context",
                result.bm25Candidates,
                result.questions,
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


def query_questions(
    *,
    query_text: str,
    top_k: int,
    round_type: RoundType,
    skill: str,
    store: MilvusQuestionStore,
) -> QueryInterviewQuestionsResult:
    try:
        vector = embed_query_text(query_text)
        candidates = store.search(
            vector=vector,
            top_k=max(VECTOR_RECALL_TOP_K, top_k),
            round_type=round_type,
        ).questions
    except Exception:
        candidates = []
    bm25_candidates = bm25_rerank_questions(candidates, query_text=query_text)[
        :BM25_RERANK_TOP_K
    ]
    questions = _sample_questions(bm25_candidates, top_k)
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


def _trace(
    round_type: RoundType,
    skill: str,
    query: str,
    log_context: str,
    candidates: list[InterviewQuestionCandidate],
    selected_questions: list[InterviewQuestionCandidate],
) -> RagRecallTrace:
    scored_candidates = _build_bm25_scored_candidates(candidates, query)
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


def _sample_questions(
    candidates: list[InterviewQuestionCandidate],
    top_k: int,
) -> list[InterviewQuestionCandidate]:
    if not candidates or top_k <= 0:
        return []
    sample_size = min(top_k, len(candidates))
    return random.sample(candidates, sample_size)


def _trace_candidate(entry: dict, final_selection_rank: int | None) -> dict:
    question: InterviewQuestionCandidate = entry["question"]
    return {
        "id": question.id,
        "questionText": question.text,
        "vectorScore": round(float(entry["vectorScore"]), 4),
        "bm25Score": round(float(entry["bm25Score"]), 4),
        "hybridScore": round(float(entry["hybridScore"]), 4),
        "matchedSkillArea": entry["matchedSkillArea"],
        "rerankRank": final_selection_rank,
        "finalSelectionRank": final_selection_rank,
        "filterReason": "selected" if final_selection_rank is not None else "not-selected",
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
