from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.question_metadata import format_skill_area, normalize_skill_area_from_text
from app.domain.question_planner import ProfessionalQuestionPlan
from app.domain.question_query import (
    RetrievalQueryIntent,
    build_professional_requeries,
    build_professional_skill_query,
    build_project_experience_query,
    describe_professional_plan_skill,
)
from app.integrations.embeddings import embed_query_text
from app.integrations.milvus_store import MilvusQuestionStore
from app.schemas.interview_state import InterviewQuestionCandidate, RoundType

VECTOR_RECALL_TOP_K = 20
BM25_RERANK_TOP_K = 5
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


def query_questions_multi(
    *,
    query_intents: list[RetrievalQueryIntent],
    top_k: int,
    round_type: RoundType,
    skill: str,
    store: MilvusQuestionStore,
) -> QueryInterviewQuestionsResult:
    if not query_intents:
        return QueryInterviewQuestionsResult(count=0, questions=[], bm25Candidates=[])

    ranked_lists: list[list[InterviewQuestionCandidate]] = []
    for intent in query_intents:
        try:
            vector = embed_query_text(intent.query)
            ranked_lists.append(
                store.search(
                    vector=vector,
                    top_k=max(VECTOR_RECALL_TOP_K, top_k),
                    round_type=round_type,
                ).questions
            )
        except Exception:
            ranked_lists.append([])

    query_text = "\n\n".join(f"[{intent.type}]\n{intent.query}" for intent in query_intents)
    fused = _rrf_merge_ranked_candidates(ranked_lists)
    reranked = metadata_rerank_questions(fused, query_text=query_text)[:BM25_RERANK_TOP_K]
    questions = _sample_questions(reranked, top_k)
    return QueryInterviewQuestionsResult(
        count=len(questions),
        questions=questions,
        bm25Candidates=reranked,
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
) -> list[InterviewQuestionCandidate]:
    scored = _build_metadata_scored_candidates(candidates, query_text)
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


def _rrf_merge_ranked_candidates(
    ranked_lists: list[list[InterviewQuestionCandidate]],
) -> list[InterviewQuestionCandidate]:
    by_id: dict[str, InterviewQuestionCandidate] = {}
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, question in enumerate(ranked, start=1):
            by_id.setdefault(question.id, question)
            scores[question.id] = scores.get(question.id, 0.0) + 1 / (RRF_K + rank)

    return sorted(
        by_id.values(),
        key=lambda question: (scores.get(question.id, 0), question.score, question.id),
        reverse=True,
    )


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


def _build_metadata_scored_candidates(
    candidates: list[InterviewQuestionCandidate],
    query_text: str,
) -> list[dict]:
    bm25_entries = _build_bm25_scored_candidates(candidates, query_text)
    if not bm25_entries:
        return []

    bm25_values = [float(entry["bm25Score"]) for entry in bm25_entries]
    vector_values = [float(entry["vectorScore"]) for entry in bm25_entries]
    max_bm25 = max(bm25_values) or 1
    min_vector = min(vector_values)
    max_vector = max(vector_values)

    scored: list[dict] = []
    for entry in bm25_entries:
        question: InterviewQuestionCandidate = entry["question"]
        retrieval_score = (
            _normalize_score(float(entry["vectorScore"]), min_vector, max_vector) * 0.45
            + (float(entry["bm25Score"]) / max_bm25) * 0.55
        )
        skill_score = _metadata_skill_match_score(question, query_text)
        job_score = _metadata_job_match_score(question, query_text)
        question_type_score = _question_type_score(question.questionType)
        difficulty_score = _difficulty_match_score(question.difficulty, query_text)
        novelty_score = 1.0
        hybrid_score = (
            0.25 * retrieval_score
            + 0.25 * skill_score
            + 0.20 * job_score
            + 0.10 * question_type_score
            + 0.10 * difficulty_score
            + 0.10 * novelty_score
        )
        scored.append({**entry, "hybridScore": hybrid_score})

    scored.sort(
        key=lambda entry: (
            entry["hybridScore"],
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
    if normalized in {"system_design", "experience_probe", "case_analysis", "scenario"}:
        return 1.0
    if normalized in {"knowledge_check", "knowledge-check"}:
        return 0.65
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
    if not candidates or top_k <= 0:
        return []
    return candidates[:top_k]


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
