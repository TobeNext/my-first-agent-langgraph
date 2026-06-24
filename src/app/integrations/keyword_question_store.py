from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.schemas.interview_state import InterviewQuestionCandidate, RoundType


@dataclass(frozen=True)
class KeywordQuestionSearchResult:
    questions: list[InterviewQuestionCandidate]


class KeywordQuestionStore(Protocol):
    def search(
        self,
        *,
        query_text: str,
        top_k: int,
        round_type: RoundType,
    ) -> KeywordQuestionSearchResult: ...


class EmptyKeywordQuestionStore:
    def search(
        self,
        *,
        query_text: str,
        top_k: int,
        round_type: RoundType,
    ) -> KeywordQuestionSearchResult:
        return KeywordQuestionSearchResult(questions=[])
