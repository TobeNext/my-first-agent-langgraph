from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagCandidate:
    id: str
    score: float
    skill_areas: tuple[str, ...] = ()


def evaluate_rag_case(
    case: dict[str, Any],
    candidates: list[RagCandidate],
    *,
    top_k: int,
) -> dict[str, Any]:
    selected = candidates[:top_k]
    selected_ids = [candidate.id for candidate in selected]
    expected_ids = [str(item) for item in case.get("expected_question_ids", [])]
    negative_ids = [str(item) for item in case.get("negative_question_ids", [])]
    acceptable_skill_areas = [
        _normalize_skill_area(item) for item in case.get("acceptable_skill_areas", [])
    ]
    expected_set = set(expected_ids)
    negative_set = set(negative_ids)

    return {
        "case_id": case["case_id"],
        "top_k": top_k,
        "candidate_count": len(candidates),
        "selected_candidate_ids": selected_ids,
        "hit_rate_at_k": _hit_rate_at_k(selected_ids, expected_set),
        "mrr": _mrr(selected_ids, expected_set),
        "ndcg": _ndcg(selected_ids, expected_ids),
        "negative_question_exclusion": _negative_question_exclusion(
            selected_ids,
            negative_set,
        ),
        "skill_area_coverage": _skill_area_coverage(selected, acceptable_skill_areas),
        "rerank_top_k_stability": _rerank_top_k_stability(selected),
        "hit_explanation": _hit_explanation(selected_ids, expected_set, negative_set),
    }


def aggregate_rag_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "hit_rate_at_k",
        "mrr",
        "ndcg",
        "negative_question_exclusion",
        "skill_area_coverage",
        "rerank_top_k_stability",
    ]
    return {
        "case_count": len(case_results),
        "metrics": {
            metric_name: _average(
                [float(result[metric_name]) for result in case_results]
            )
            for metric_name in metric_names
        },
        "cases": case_results,
    }


def candidate_from_mapping(value: dict[str, Any]) -> RagCandidate:
    return RagCandidate(
        id=str(value["id"]),
        score=float(value.get("score", 0.0)),
        skill_areas=tuple(str(item) for item in value.get("skill_areas", [])),
    )


def _hit_rate_at_k(selected_ids: list[str], expected_set: set[str]) -> float:
    if not expected_set:
        return 0.0
    return len(expected_set & set(selected_ids)) / len(expected_set)


def _mrr(selected_ids: list[str], expected_set: set[str]) -> float:
    for rank, candidate_id in enumerate(selected_ids, start=1):
        if candidate_id in expected_set:
            return 1 / rank
    return 0.0


def _ndcg(selected_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    dcg = 0.0
    for rank, candidate_id in enumerate(selected_ids, start=1):
        relevance = 1.0 if candidate_id in expected_set else 0.0
        dcg += relevance / math.log2(rank + 1)
    ideal_hits = min(len(expected_set), len(selected_ids))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return _round(dcg / ideal_dcg) if ideal_dcg else 0.0


def _negative_question_exclusion(selected_ids: list[str], negative_set: set[str]) -> float:
    if not negative_set:
        return 1.0
    leaked = negative_set & set(selected_ids)
    return _round(1 - (len(leaked) / len(negative_set)))


def _skill_area_coverage(
    candidates: list[RagCandidate],
    acceptable_skill_areas: list[str],
) -> float:
    if not acceptable_skill_areas:
        return 1.0
    selected_skill_areas = {
        _normalize_skill_area(skill_area)
        for candidate in candidates
        for skill_area in candidate.skill_areas
    }
    covered = [
        skill_area
        for skill_area in acceptable_skill_areas
        if any(
            skill_area in selected or selected in skill_area
            for selected in selected_skill_areas
        )
    ]
    return _round(len(covered) / len(acceptable_skill_areas))


def _rerank_top_k_stability(candidates: list[RagCandidate]) -> float:
    if len(candidates) <= 1:
        return 1.0
    sorted_candidates = sorted(candidates, key=lambda item: (-item.score, item.id))
    stable_positions = sum(
        1
        for actual, expected in zip(candidates, sorted_candidates, strict=True)
        if actual.id == expected.id
    )
    return _round(stable_positions / len(candidates))


def _hit_explanation(
    selected_ids: list[str],
    expected_set: set[str],
    negative_set: set[str],
) -> dict[str, list[str]]:
    return {
        "matched_expected_ids": [
            candidate_id for candidate_id in selected_ids if candidate_id in expected_set
        ],
        "missed_expected_ids": sorted(expected_set - set(selected_ids)),
        "leaked_negative_ids": [
            candidate_id for candidate_id in selected_ids if candidate_id in negative_set
        ],
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _normalize_skill_area(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", " ").split())


def _round(value: float) -> float:
    return round(value, 4)
