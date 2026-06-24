from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pymilvus import MilvusClient

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import get_settings
from app.domain.question_metadata import (
    format_skill_area,
    normalize_difficulty_score,
    normalize_level,
    normalize_question_type,
    normalize_skill_area_from_metadata,
)
from app.schemas.interview_state import InterviewQuestionCandidate, RoundType

logger = logging.getLogger(__name__)
ROUND_ROLE_VALUES = {"professional-skills", "project-experience"}
GLOBAL_QUESTION_USER_ID = "global"
DEFAULT_QUESTION_LANGUAGE = "zh"

_DEFAULT_HNSW_M = 16
_DEFAULT_HNSW_EF_CONSTRUCTION = 200
_DEFAULT_HNSW_EF_SEARCH = 64
_DEFAULT_POOL_SIZE = 4


@dataclass(frozen=True)
class MilvusSearchResult:
    questions: list[InterviewQuestionCandidate]


class MilvusQuestionStore:
    def __init__(
        self,
        *,
        address: str | None = None,
        collection_name: str = "interview_questions",
        pool_size: int = _DEFAULT_POOL_SIZE,
    ):
        self.address = _normalize_milvus_uri(address or get_settings().milvus_address)
        self.collection_name = collection_name
        self._pool_size = pool_size
        self._client: MilvusClient | None = None

    @property
    def client(self) -> MilvusClient:
        """Lazily create and reuse a pooled MilvusClient singleton."""
        if self._client is None:
            self._client = MilvusClient(uri=self.address, pool_size=self._pool_size)
        return self._client

    def close(self) -> None:
        """Release the MilvusClient connection pool."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Error closing MilvusClient; ignoring.", exc_info=True)
            finally:
                self._client = None

    def ensure_hnsw_index(
        self,
        *,
        metric_type: str = "COSINE",
        m: int = _DEFAULT_HNSW_M,
        ef_construction: int = _DEFAULT_HNSW_EF_CONSTRUCTION,
    ) -> bool:
        """Create or overwrite HNSW index on the vector field.

        Returns True if the index was created/updated, False if not applicable
        (e.g. collection does not exist or no vector field found).
        """
        try:
            if not self.client.has_collection(self.collection_name):
                logger.warning(
                    "Cannot create HNSW index: collection %s does not exist.",
                    self.collection_name,
                )
                return False
        except Exception as exc:
            logger.warning(
                "Failed to check collection existence for index creation. collection=%s error=%s",
                self.collection_name,
                exc,
            )
            return False

        description = self.client.describe_collection(self.collection_name)
        fields = description.get("fields") if isinstance(description, dict) else []
        vector_field = None
        for field in fields:
            if isinstance(field, dict) and isinstance(field.get("name"), str):
                if field.get("type") in ("FLOAT_VECTOR", 100, 101):
                    vector_field = str(field["name"])
                    break

        if vector_field is None:
            logger.warning(
                "Cannot create HNSW index: no vector field found in collection %s.",
                self.collection_name,
            )
            return False

        try:
            self.client.create_index(
                collection_name=self.collection_name,
                field_name=vector_field,
                index_params={
                    "index_type": "HNSW",
                    "metric_type": metric_type,
                    "params": {
                        "M": m,
                        "efConstruction": ef_construction,
                    },
                },
            )
            logger.info(
                "HNSW index created on collection=%s field=%s metric=%s M=%d efConstruction=%d",
                self.collection_name,
                vector_field,
                metric_type,
                m,
                ef_construction,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to create HNSW index. collection=%s field=%s error=%s",
                self.collection_name,
                vector_field,
                exc,
            )
            return False

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        round_type: RoundType,
    ) -> MilvusSearchResult:
        with _get_tracer().start_as_current_span(
            "milvus.question_retrieval.search",
            attributes={
                "db.system": "milvus",
                "db.collection.name": self.collection_name,
                "rag.top_k": top_k,
                "interview.round_type": round_type,
            },
        ) as span:
            try:
                client = self.client
                fields = _collection_fields(client, self.collection_name)
                has_filter_fields = {"isActive", "language", "userId"}.issubset(fields)
                output_fields = ["id", "metadata"]
                for field in (
                    "role",
                    "difficulty",
                    "skillArea",
                    "language",
                    "isActive",
                    "userId",
                ):
                    if field in fields:
                        output_fields.append(field)

                span.set_attribute("db.milvus.scalar_filter", has_filter_fields)
                rows = client.search(
                    collection_name=self.collection_name,
                    data=[vector],
                    limit=top_k,
                    anns_field="vector",
                    output_fields=output_fields,
                    search_params={
                        "metric_type": "COSINE",
                        "params": {"ef": _DEFAULT_HNSW_EF_SEARCH},
                    },
                    filter=(
                        'isActive == true and language == "zh" and userId == "global"'
                        if has_filter_fields
                        else None
                    ),
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("rag.result_count", 0)
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
                if not _passes_metadata_filter(entity, metadata):
                    continue
                role = str(raw_role or round_type).strip() or round_type
                question_text = str(
                    metadata.get("question") or entity.get("question") or ""
                ).strip()
                questions.append(
                    InterviewQuestionCandidate.model_validate(
                        {
                            "id": str(
                                entity.get("id")
                                or row.get("id")
                                or f"milvus-{len(questions) + 1}"
                            ),
                            "text": question_text,
                            "score": float(row.get("distance") or row.get("score") or 0),
                            "role": role,
                            "questionType": _question_type(metadata),
                            "difficulty": normalize_difficulty_score(
                                entity.get("difficulty")
                                or metadata.get("difficulty")
                                or metadata.get("level")
                            ),
                            "skillArea": _skill_area(
                                entity.get("skillArea")
                                or metadata.get("skillArea")
                                or metadata.get("skills")
                            )
                            or normalize_skill_area_from_metadata(
                                {
                                    **metadata,
                                    "question": question_text,
                                    "role": entity.get("role") or metadata.get("role"),
                                }
                            ),
                            "answer": metadata.get("answer"),
                            "tags": _tags(metadata.get("tags")),
                            "answerPoints": _string_list(
                                metadata.get("answer_points") or metadata.get("answerPoints")
                            ),
                            "skills": _skill_area(
                                metadata.get("skills")
                                or entity.get("skillArea")
                                or metadata.get("skillArea")
                            ),
                            "level": normalize_level(
                                metadata.get("level") or metadata.get("difficulty")
                            ),
                            "jobFamily": metadata.get("job_family") or metadata.get("jobFamily"),
                            "jobDuties": _string_list(
                                metadata.get("job_duties") or metadata.get("jobDuties")
                            ),
                            "language": str(
                                entity.get("language")
                                or metadata.get("language")
                                or DEFAULT_QUESTION_LANGUAGE
                            ).strip()
                            or DEFAULT_QUESTION_LANGUAGE,
                            "embeddingText": metadata.get("embedding_text")
                            or metadata.get("embeddingText"),
                            "source": metadata.get("source"),
                            "isActive": _bool_or_default(
                                entity.get("isActive", metadata.get("isActive")),
                                default=True,
                            ),
                            "userId": str(
                                entity.get("userId")
                                or metadata.get("userId")
                                or GLOBAL_QUESTION_USER_ID
                            ).strip()
                            or GLOBAL_QUESTION_USER_ID,
                        }
                    )
                )
            span.set_attribute("rag.result_count", len(questions))
            return MilvusSearchResult(questions=questions)

    def collection_exists(self) -> bool:
        try:
            return bool(self.client.has_collection(self.collection_name))
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


def _question_type(metadata: dict[str, Any]) -> str:
    raw = metadata.get("question_type") or metadata.get("questionType")
    return normalize_question_type(raw)


def _legacy_difficulty(value: Any) -> str | None:
    level = normalize_level(value)
    if level == "senior":
        return "hard"
    if level == "junior":
        return "easy"
    if level == "middle":
        return "medium"
    return None


def _passes_metadata_filter(entity: dict[str, Any], metadata: dict[str, Any]) -> bool:
    question = str(metadata.get("question") or entity.get("question") or "").strip()
    answer = str(metadata.get("answer") or entity.get("answer") or "").strip()
    text = str(metadata.get("text") or entity.get("text") or "").strip()
    if not question or not answer or not text:
        return False

    if not _bool_or_default(entity.get("isActive", metadata.get("isActive")), default=True):
        return False
    language = str(
        entity.get("language") or metadata.get("language") or DEFAULT_QUESTION_LANGUAGE
    ).strip()
    if language != DEFAULT_QUESTION_LANGUAGE:
        return False
    user_id = str(
        entity.get("userId") or metadata.get("userId") or GLOBAL_QUESTION_USER_ID
    ).strip()
    return user_id == GLOBAL_QUESTION_USER_ID


def _bool_or_default(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _string_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        return result or None
    if isinstance(value, str) and value.strip():
        result = [item.strip() for item in value.split(",") if item.strip()]
        return result or None
    return None


def _get_tracer() -> trace.Tracer:
    return trace.get_tracer("interview-python-agent")
