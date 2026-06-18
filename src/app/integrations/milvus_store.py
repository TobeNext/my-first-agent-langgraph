from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.domain.question_metadata import format_skill_area, normalize_skill_area_from_metadata
from app.schemas.interview_state import InterviewQuestionCandidate, RoundType

logger = logging.getLogger(__name__)
ROUND_ROLE_VALUES = {"professional-skills", "project-experience"}


@dataclass(frozen=True)
class MilvusSearchResult:
    questions: list[InterviewQuestionCandidate]


class MilvusQuestionStore:
    def __init__(self, *, address: str | None = None, collection_name: str = "interview_questions"):
        self.address = _normalize_milvus_uri(address or get_settings().milvus_address)
        self.collection_name = collection_name

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        round_type: RoundType,
    ) -> MilvusSearchResult:
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=self.address)
            fields = _collection_fields(client, self.collection_name)
            has_scalar_fields = {"role", "difficulty", "skillArea"}.issubset(fields)
            output_fields = ["id", "metadata"]
            if has_scalar_fields:
                output_fields.extend(["role", "difficulty", "skillArea"])

            rows = client.search(
                collection_name=self.collection_name,
                data=[vector],
                limit=top_k,
                anns_field="vector",
                output_fields=output_fields,
                filter=f'role == "{round_type}"' if has_scalar_fields else None,
            )
        except Exception as exc:
            logger.warning(
                "Milvus interview question search failed; falling back to empty recall. "
                "collection=%s address=%s round_type=%s error=%s",
                self.collection_name,
                self.address,
                round_type,
                exc,
            )
            return MilvusSearchResult(questions=[])

        questions: list[InterviewQuestionCandidate] = []
        for row in rows[0] if rows else []:
            entity = _entity(row)
            metadata = _metadata(entity.get("metadata"))
            raw_role = entity.get("role") or metadata.get("role")
            if not _matches_round_type(raw_role, round_type):
                continue
            role = raw_role if raw_role in ROUND_ROLE_VALUES else round_type
            text = str(
                metadata.get("question") or metadata.get("text") or entity.get("text") or ""
            ).strip()
            if not text:
                continue
            questions.append(
                InterviewQuestionCandidate.model_validate(
                    {
                        "id": str(
                            entity.get("id") or row.get("id") or f"milvus-{len(questions) + 1}"
                        ),
                        "text": text,
                        "score": float(row.get("distance") or row.get("score") or 0),
                        "role": role,
                        "difficulty": entity.get("difficulty") or metadata.get("difficulty"),
                        "skillArea": _skill_area(
                            entity.get("skillArea") or metadata.get("skillArea")
                        )
                        or normalize_skill_area_from_metadata(
                            {
                                **metadata,
                                "question": text,
                                "role": entity.get("role") or metadata.get("role"),
                            }
                        ),
                        "answer": metadata.get("answer"),
                        "tags": _tags(metadata.get("tags")),
                    }
                )
            )
        return MilvusSearchResult(questions=questions)

    def collection_exists(self) -> bool:
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=self.address)
            return bool(client.has_collection(self.collection_name))
        except Exception as exc:
            logger.warning(
                "Milvus collection existence check failed. collection=%s address=%s error=%s",
                self.collection_name,
                self.address,
                exc,
            )
            return False


def _normalize_milvus_uri(address: str) -> str:
    value = address.strip()
    if value.startswith(("http://", "https://", "tcp://", "unix://")) or value.endswith(".db"):
        return value
    return f"http://{value}"


def _collection_fields(client: Any, collection_name: str) -> set[str]:
    description = client.describe_collection(collection_name)
    raw_fields = description.get("fields") if isinstance(description, dict) else None
    if not isinstance(raw_fields, list):
        schema = description.get("schema") if isinstance(description, dict) else None
        raw_fields = schema.get("fields") if isinstance(schema, dict) else []

    fields: set[str] = set()
    for field in raw_fields:
        if isinstance(field, dict) and isinstance(field.get("name"), str):
            fields.add(field["name"])
    return fields


def _matches_round_type(role: Any, round_type: RoundType) -> bool:
    if role in ROUND_ROLE_VALUES:
        return role == round_type
    return round_type == "professional-skills"


def _entity(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        raw = row.get("entity") or row
        return raw if isinstance(raw, dict) else {}
    raw = getattr(row, "entity", None)
    return raw if isinstance(raw, dict) else {}


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _skill_area(value: Any) -> list[str] | None:
    return format_skill_area(value) or None


def _tags(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return None
